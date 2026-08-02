"""
test_chat_ui.py — Tests for the chat UI's non-Streamlit logic.

Run from the thesis-code root, ideally in the pipeline venv:

    source regulatory_testgen/venv/bin/activate
    python -m unittest chat_ui.tests.test_chat_ui -v

The Streamlit script (app.py) can't be imported without a running Streamlit
runtime, so we test the pieces that carry the real logic: the clause↔PDF
correlation and highlight rendering (pdf_highlight), the clause index shape
(retrieval), and — most importantly for the "wrong citations" bug — the rule
that ONLY read/generated clauses become citations, never search candidates
(tools). See test_chat_ui_ui_review.md for the UI/design review.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CHAT_UI = _ROOT / "chat_ui"
for _p in (_CHAT_UI, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pdf_highlight  # noqa: E402


# ── Synthetic middle.json builder ────────────────────────────────────────────

def _block(text: str, bbox):
    return {
        "bbox": bbox,
        "lines": [{"spans": [{"type": "text", "content": text}]}],
    }


def _write_middle(tmpdir: Path, pages) -> Path:
    """pages: list of (page_idx, [(text, bbox), ...]) → writes a middle.json."""
    pdf_info = []
    for page_idx, blocks in pages:
        pdf_info.append({
            "page_idx": page_idx,
            "page_size": [595, 842],
            "para_blocks": [_block(t, b) for t, b in blocks],
        })
    path = tmpdir / "doc_middle.json"
    path.write_text(json.dumps({"pdf_info": pdf_info}), encoding="utf-8")
    return path


class _FakeClause:
    def __init__(self, clause_id, title, text, is_pseudo=False):
        self.clause_id = clause_id
        self.title = title
        self.text = text
        self.is_pseudo_clause = is_pseudo


# ── pdf_highlight: pure helpers ──────────────────────────────────────────────

class TestPdfHighlightHelpers(unittest.TestCase):
    def test_normalize_collapses_whitespace_and_lowercases(self):
        self.assertEqual(pdf_highlight._normalize("  Hello\n  WORLD \t"), "hello world")
        self.assertEqual(pdf_highlight._normalize(None), "")

    def test_boxes_by_page_groups_and_skips_unknown(self):
        page_map = {
            "5.1": [{"page": 0, "bbox": [0, 0, 1, 1]}, {"page": 2, "bbox": [1, 1, 2, 2]}],
            "5.2": [{"page": 0, "bbox": [3, 3, 4, 4]}],
        }
        boxes = pdf_highlight._boxes_by_page(["5.1", "5.2", "missing"], page_map)
        self.assertEqual(sorted(boxes), [0, 2])
        self.assertEqual(len(boxes[0]), 2)  # 5.1 + 5.2 both on page 0
        self.assertEqual(len(boxes[2]), 1)

    def test_flatten_middle_blocks_reads_span_text(self):
        with tempfile.TemporaryDirectory() as d:
            mid = _write_middle(Path(d), [
                (0, [("First block of text here", [10, 10, 200, 30])]),
                (1, [("Second block on page two", [10, 10, 200, 30])]),
            ])
            blocks = pdf_highlight._flatten_middle_blocks(mid)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["page"], 0)
        self.assertIn("First block", blocks[0]["text"])


# ── pdf_highlight: correlation algorithm ─────────────────────────────────────

class TestBuildClausePageMap(unittest.TestCase):
    def test_matches_clauses_in_document_order(self):
        with tempfile.TemporaryDirectory() as d:
            mid = _write_middle(Path(d), [
                (0, [
                    ("The system shall provide a collision warning to the driver", [10, 10, 400, 30]),
                    ("The vehicle must decelerate to avoid the obstacle ahead", [10, 40, 400, 60]),
                ]),
                (1, [
                    ("Testing conditions require a dry paved road surface", [10, 10, 400, 30]),
                ]),
            ])
            clauses = [
                _FakeClause("5.1", "Warning", "The system shall provide a collision warning to the driver"),
                _FakeClause("5.2", "Braking", "The vehicle must decelerate to avoid the obstacle ahead"),
                _FakeClause("6.1", "Road", "Testing conditions require a dry paved road surface"),
            ]
            page_map = pdf_highlight.build_clause_page_map(clauses, mid)

        self.assertIn("5.1", page_map)
        self.assertEqual(page_map["5.1"][0]["page"], 0)
        self.assertEqual(page_map["5.2"][0]["page"], 0)
        self.assertEqual(page_map["6.1"][0]["page"], 1)

    def test_pseudo_clauses_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            mid = _write_middle(Path(d), [
                (0, [("The system shall provide a collision warning to the driver", [10, 10, 400, 30])]),
            ])
            clauses = [
                _FakeClause("TOC", "Contents", "The system shall provide a collision warning to the driver", is_pseudo=True),
            ]
            page_map = pdf_highlight.build_clause_page_map(clauses, mid)
        self.assertNotIn("TOC", page_map)

    def test_short_targets_are_not_matched(self):
        with tempfile.TemporaryDirectory() as d:
            mid = _write_middle(Path(d), [
                (0, [("The system shall provide a collision warning to the driver", [10, 10, 400, 30])]),
            ])
            clauses = [_FakeClause("9.9", "x", "short")]  # below _MIN_MATCH_LEN
            page_map = pdf_highlight.build_clause_page_map(clauses, mid)
        self.assertNotIn("9.9", page_map)

    def test_missing_middle_json_returns_empty(self):
        self.assertEqual(pdf_highlight.build_clause_page_map([], Path("/no/such/file.json")), {})

    def test_contains_matches_either_direction(self):
        # block ⊆ clause (parser concatenated blocks into one clause)
        self.assertTrue(pdf_highlight._contains(
            "the system shall brake", "the system shall brake automatically when needed"))
        # clause ⊆ block (MinerU kept the "5.1.1." number prefix the parser stripped)
        self.assertTrue(pdf_highlight._contains(
            "5.1.1. the system shall brake automatically", "the system shall brake automatically"))
        # unrelated
        self.assertFalse(pdf_highlight._contains(
            "completely different sentence here", "the system shall brake automatically"))
        # too-short block never matches
        self.assertFalse(pdf_highlight._contains("shall", "the system shall brake automatically"))

    def test_number_prefixed_blocks_still_match(self):
        """Regression: MinerU prepends the clause number to the block text, so
        the clause text is a substring of the block, not vice-versa."""
        with tempfile.TemporaryDirectory() as d:
            mid = _write_middle(Path(d), [
                (0, [
                    ("5.1.1. any vehicle fitted with an aebs shall meet the performance requirements",
                     [10, 10, 400, 30]),
                ]),
            ])
            clauses = [_FakeClause(
                "5.1.1", "", "any vehicle fitted with an aebs shall meet the performance requirements")]
            page_map = pdf_highlight.build_clause_page_map(clauses, mid)
        self.assertIn("5.1.1", page_map)


# ── pdf_highlight: rendering (needs PyMuPDF; skips if absent) ─────────────────

def _has_fitz():
    try:
        import fitz  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_fitz(), "PyMuPDF (fitz) not installed")
class TestHighlightRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fitz
        cls.tmp = tempfile.TemporaryDirectory()
        p = Path(cls.tmp.name) / "sample.pdf"
        doc = fitz.open()
        for _ in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), "Sample regulatory text on this page.")
        doc.save(str(p))
        doc.close()
        cls.pdf_path = p
        # clause "A" is on page 1 (0-indexed), clause "B" unlocated
        cls.page_map = {"A": [{"page": 1, "bbox": [70, 65, 300, 85]}]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_page_images_are_valid_png(self):
        imgs = pdf_highlight.render_highlighted_page_images(self.pdf_path, ["A"], self.page_map)
        self.assertEqual(len(imgs), 1)
        page_no, png = imgs[0]
        self.assertEqual(page_no, 2)  # 1-indexed
        self.assertTrue(png.startswith(b"\x89PNG"), "should be PNG bytes")

    def test_unlocated_clause_yields_no_images(self):
        self.assertEqual(pdf_highlight.render_highlighted_page_images(self.pdf_path, ["B"], self.page_map), [])

    def test_pdf_is_valid_and_trimmed(self):
        result = pdf_highlight.render_highlighted_pdf(self.pdf_path, ["A"], self.page_map)
        self.assertIsNotNone(result)
        pdf_bytes, pages = result
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertEqual(pages, [2])  # only the one cited page, 1-indexed

    def test_source_pdf_unmodified_on_disk(self):
        before = self.pdf_path.read_bytes()
        pdf_highlight.render_highlighted_page_images(self.pdf_path, ["A"], self.page_map)
        pdf_highlight.render_highlighted_pdf(self.pdf_path, ["A"], self.page_map)
        self.assertEqual(self.pdf_path.read_bytes(), before, "original PDF must stay byte-identical")


# ── Citation recording rules (the core "wrong citations" fix) ────────────────

class _DummyPlaceholder:
    def info(self, *a, **k):
        return self

    def empty(self, *a, **k):
        return self

    def write(self, *a, **k):
        return self


class _SessionState(dict):
    """dict that also allows attribute access, like Streamlit's session_state."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _FakeSt:
    """Minimal stand-in for the streamlit module used inside tools.py."""

    def __init__(self):
        self.session_state = _SessionState()

    def empty(self):
        return _DummyPlaceholder()


class TestCitationRecording(unittest.TestCase):
    def setUp(self):
        import tools
        self.tools = tools
        self.fake_st = _FakeSt()
        self._orig_st = tools.st
        tools.st = self.fake_st
        self.clause_index = {
            "5.2.1": {"title": "Warning", "text_preview": "shall warn", "category": "obligation"},
            "6.1": {"title": "Road", "text_preview": "dry road", "category": "test_condition"},
        }

    def tearDown(self):
        self.tools.st = self._orig_st

    def _dispatch(self, name, args, **kw):
        return self.tools.dispatch_tool(
            name, args,
            graph=kw.get("graph"),
            llm_config=kw.get("llm_config", object()),
            clause_map=kw.get("clause_map", {}),
            clause_index=self.clause_index,
            checkpoints_dir=kw.get("checkpoints_dir"),
        )

    def test_search_candidates_are_not_citations(self):
        import retrieval
        orig = retrieval.find_relevant_clauses
        retrieval.find_relevant_clauses = lambda *a, **k: ["5.2.1", "6.1"]
        try:
            self._dispatch("search_clauses", {"query": "braking"})
        finally:
            retrieval.find_relevant_clauses = orig
        # Search results are candidates, not citations:
        self.assertEqual(self.fake_st.session_state.get("last_retrieved_clause_ids", []), [])
        self.assertEqual(self.fake_st.session_state.get("last_search_candidates"), ["5.2.1", "6.1"])

    def test_get_clause_is_a_citation(self):
        self._dispatch("get_clause", {"clause_id": "5.2.1"})
        self.assertEqual(self.fake_st.session_state["last_retrieved_clause_ids"], ["5.2.1"])

    def test_get_performance_table_is_a_citation(self):
        class _Graph:
            def get_tables(self, cid):
                return []
        self._dispatch("get_performance_table", {"clause_id": "6.1"}, graph=_Graph())
        self.assertEqual(self.fake_st.session_state["last_retrieved_clause_ids"], ["6.1"])

    def test_document_structure_produces_no_citations(self):
        # This is the reported bug: a "what is this about" overview must not cite.
        self._dispatch("get_document_structure", {})
        self.assertEqual(self.fake_st.session_state.get("last_retrieved_clause_ids", []), [])

    def test_generate_cites_real_source_clause_ids(self):
        import generator

        class _TC:
            def __init__(self, srcs):
                self._srcs = srcs

            def model_dump(self):
                return {"test_id": "T1", "source_clause_ids": self._srcs}

        orig = generator.generate_for_clause
        # request was for 5.2.1, but the ReAct agent actually drew on 5.2.1 + 6.1
        generator.generate_for_clause = lambda cid, *a, **k: [_TC(["5.2.1", "6.1"])]
        try:
            self._dispatch("generate_test_cases", {"clause_ids": ["5.2.1"]})
        finally:
            generator.generate_for_clause = orig
        self.assertEqual(
            sorted(self.fake_st.session_state["last_retrieved_clause_ids"]),
            ["5.2.1", "6.1"],
        )
        self.assertEqual(len(self.fake_st.session_state["pending_results"]), 1)


# ── Generation context: parent section + referenced clauses ─────────────────

class _GClause:
    """Minimal stand-in for a parsed Clause, sufficient for RegulatoryGraph.build."""

    def __init__(self, cid, title="", text="", refs=None):
        self.clause_id = cid
        self.title = title
        self.text = text
        self.references = refs or []
        self.is_pseudo_clause = False
        self.document_region = "body"
        self.section_path = [cid]


def _has_graph_deps():
    try:
        from regulatory_testgen.graph import RegulatoryGraph  # noqa: F401
        return True
    except Exception:
        return False


def _build_sample_graph():
    """5 ⊃ 5.2 ⊃ 5.2.1(→6.1); 6.1 and 6.2 are separate sections."""
    from regulatory_testgen.graph import RegulatoryGraph
    clauses = [
        _GClause("5", "General", "General requirements for the system."),
        _GClause("5.2", "Car to car", "Requirements for car-to-car scenarios."),
        _GClause("5.2.1", "Stationary target",
                 "The vehicle shall detect a stationary target ahead.", refs=["6.1"]),
        _GClause("6.1", "Test conditions", "A dry paved road surface is required."),
        _GClause("6.2", "Unrelated setup", "Completely unrelated content here."),
    ]
    return RegulatoryGraph.build(clauses, [])


def _biref_fixture():
    """Fixture exercising fmt_root_biref. Structure + categories + a clause_index
    mirror, so the graph and the clause_index adapter can be checked for parity.

      5   Requirements            [formatting]
      5.1 General requirements    [formatting]  -> 5.1.1 obligation
      5.2 Car to car scenario     [formatting]  -> 5.2.1 (seed, obligation, refs 6.1)
                                                   5.2.2 obligation
      6   Test                    [formatting]  -> 6.1 test_condition
                                                   6.2 test_condition, refs 5.2.1  (INTO the block)
      7   Unrelated obligation
    The win: 6.2 references INTO the 5.2 block, so only the *incoming* direction
    of the block's ref closure reaches it."""
    from regulatory_testgen.graph import RegulatoryGraph
    specs = [
        ("5", "Requirements", "", [], "formatting"),
        ("5.1", "General requirements", "", [], "formatting"),
        ("5.1.1", "Scope", "The system applies generally.", [], "obligation"),
        ("5.2", "Car to car scenario", "", [], "formatting"),
        ("5.2.1", "Warning", "A collision warning shall be given.", ["6.1"], "obligation"),
        ("5.2.2", "Braking", "Emergency braking shall be applied.", [], "obligation"),
        ("5.3", "Robustness", "See 5.2.1; false reactions shall be avoided.", ["5.2.1"], "obligation"),
        ("6", "Test", "", [], "formatting"),
        ("6.1", "Conditions", "Dry paved road.", [], "test_condition"),
        ("6.2", "Target", "For the test of 5.2.1 a soft target is used.", ["5.2.1"], "test_condition"),
        ("7", "Unrelated", "Completely unrelated obligation here.", [], "obligation"),
    ]
    clauses = [_GClause(cid, title, text, refs) for cid, title, text, refs, _cat in specs]
    graph = RegulatoryGraph.build(clauses, [])
    categories = {cid: cat for cid, _t, _x, _r, cat in specs}
    clause_index = {
        cid: {"category": cat,
              "parent": __import__("regulatory_testgen.structure_ids",
                                   fromlist=["parent_of"]).parent_of(cid),
              "references": refs}
        for cid, _t, _x, refs, cat in specs
    }
    return graph, categories, clause_index


def _build_sibling_graph():
    """5 ⊃ 5.2 ⊃ {5.2.1 ⊃ 5.2.1.1, 5.2.2}; 6.2 is an unrelated section.

    Exercises the sibling sub-block: a test for 5.2.1 must also see its peer
    5.2.2 and its own child 5.2.1.1, neither of which is textually referenced."""
    from regulatory_testgen.graph import RegulatoryGraph
    clauses = [
        _GClause("5", "General", "General requirements."),
        _GClause("5.2", "Car to car", "Requirements for car-to-car scenarios."),
        _GClause("5.2.1", "Warning", "A collision warning shall be given."),
        _GClause("5.2.1.1", "Timing", "The warning shall be timely."),
        _GClause("5.2.2", "Braking", "Emergency braking shall be applied."),
        _GClause("6.2", "Unrelated setup", "Completely unrelated content here."),
    ]
    return RegulatoryGraph.build(clauses, [])


@unittest.skipUnless(_has_graph_deps(), "regulatory_testgen.graph deps not installed")
class TestGenerationContext(unittest.TestCase):
    def setUp(self):
        self.graph = _build_sample_graph()

    def test_ancestors_are_parent_chain_nearest_first(self):
        self.assertEqual(self.graph.get_ancestors("5.2.1"), ["5.2", "5"])
        self.assertEqual(self.graph.get_ancestors("5.2"), ["5"])
        self.assertEqual(self.graph.get_ancestors("5"), [])

    def test_generation_context_includes_parents_and_references(self):
        ids = [c.clause_id for c in self.graph.get_generation_context("5.2.1")]
        # target first, then its parent section(s) and referenced clause
        self.assertEqual(ids[0], "5.2.1")
        self.assertIn("5.2", ids)   # parent section — the reported gap
        self.assertIn("5", ids)     # grandparent
        self.assertIn("6.1", ids)   # transitive REFERS_TO
        self.assertNotIn("6.2", ids)  # unrelated section must NOT be pulled in

    def test_generation_context_dedups(self):
        ids = [c.clause_id for c in self.graph.get_generation_context("5.2.1")]
        self.assertEqual(len(ids), len(set(ids)))

    def test_generation_context_includes_sibling_sub_block(self):
        # 5.2.1 and 5.2.2 split one requirement across sibling sub-clauses; a
        # test for 5.2.1 needs 5.2.2 too, but nothing textually links them.
        g = _build_sibling_graph()
        ids = [c.clause_id for c in g.get_generation_context("5.2.1")]
        self.assertEqual(ids[0], "5.2.1")
        self.assertIn("5.2.2", ids)      # sibling sub-block (same parent 5.2)
        self.assertIn("5.2.1.1", ids)    # the seed's own descendant
        self.assertNotIn("6.2", ids)     # unrelated section still excluded

    def test_get_children_returns_direct_subclauses(self):
        g = _build_sibling_graph()
        self.assertEqual(sorted(g.get_children("5.2")), ["5.2.1", "5.2.2"])
        self.assertEqual(g.get_children("5.2.2"), [])  # leaf

    def test_fmt_root_biref_needs_categories(self):
        # Without categories attached, the formatting-root block is skipped: the
        # incoming-reference clause 6.2 must NOT be pulled in.
        g, _cats, _idx = _biref_fixture()
        ids = [c.clause_id for c in g.get_generation_context("5.2.1")]
        self.assertIn("5.2.2", ids)     # self_sub still works (no categories needed)
        self.assertNotIn("6.2", ids)    # fmt_root_biref off -> incoming ref not reached

    def test_deployed_context_is_bidirectional_excluding_testing(self):
        # Deployed retrieval follows references BOTH ways but EXCLUDES testing
        # clauses. So the incoming OBLIGATION reference 5.3 (5.3 -> 5.2.1) is
        # reached, while the testing clauses 6.1 (outgoing) and 6.2 (incoming) are
        # not — they belong to step 2.
        g, cats, _idx = _biref_fixture()
        g.set_categories(cats)
        ids = [c.clause_id for c in g.get_generation_context("5.2.1")]
        self.assertEqual(ids[0], "5.2.1")
        self.assertIn("5.2", ids)       # formatting root (kept)
        self.assertIn("5.2.2", ids)     # sibling sub-block
        self.assertIn("5.3", ids)       # INCOMING obligation ref — bidirectional win
        self.assertNotIn("6.1", ids)    # testing clause excluded (even as outgoing ref)
        self.assertNotIn("6.2", ids)    # testing clause excluded (incoming ref)
        self.assertNotIn("7", ids)      # unrelated clause stays out

    def test_bidirectional_option_without_exclusion_pulls_testing(self):
        # Without exclude_testing, the bidirectional closure reaches the testing
        # clause 6.2 (incoming) — the exclusion is what keeps it out above.
        from regulatory_testgen.context_expand import generation_context_from_index
        _g, _cats, idx = _biref_fixture()
        ids = generation_context_from_index(
            idx, "5.2.1", cap_ancestors_at_formatting_root=True,
            transitive_refs=True, bidirectional_refs=True, exclude_testing=False)
        self.assertIn("6.2", ids)

    def test_graph_and_index_adapters_agree(self):
        # The deployed graph and the validator's clause_index adapter must return
        # the SAME set (they share regulatory_testgen.context_expand): capped,
        # bidirectional, transitive, testing-excluded — same flags on both sides.
        from regulatory_testgen.context_expand import (
            generation_context_from_index, OBLIGATION_STRATEGY)
        g, cats, idx = _biref_fixture()
        g.set_categories(cats)
        from_graph = {c.clause_id for c in g.get_generation_context("5.2.1")}
        from_index = set(generation_context_from_index(idx, "5.2.1", **OBLIGATION_STRATEGY))
        self.assertEqual(from_graph, from_index)

    def test_procedure_context_collects_testing_clauses(self):
        # Step 2: procedure retrieval seeds from testing-labelled clauses.
        from regulatory_testgen.context_expand import procedure_context_from_index
        _g, _cats, idx = _biref_fixture()
        proc = set(procedure_context_from_index(idx))
        self.assertIn("6.1", proc)      # test_condition clause is collected
        self.assertIn("6.2", proc)      # test_condition clause is collected
        self.assertNotIn("7", proc)     # a plain obligation is not a procedure seed

    def test_get_test_procedures_tool_lists_testing_clauses(self):
        # The model-facing tool returns the testing clauses reachable from a
        # requirement (procedures are NOT excluded here, unlike obligation retrieval).
        from tools import handle_get_test_procedures
        _g, _cats, idx = _biref_fixture()
        out = handle_get_test_procedures("5.2.1", clause_index=idx)
        self.assertIn("6.1", out)       # test_condition reachable from 5.2.1
        self.assertIn("6.2", out)
        self.assertNotIn("5.2.2", out)  # a plain obligation is not a procedure

    def test_generation_context_caps_ancestors_at_formatting_root(self):
        # seed 5.2.1's nearest formatting ancestor is 5.2; the top section 5 is
        # above it and must be dropped now that the deployed path caps.
        g, cats, _idx = _biref_fixture()
        g.set_categories(cats)
        ids = [c.clause_id for c in g.get_generation_context("5.2.1")]
        self.assertIn("5.2", ids)       # the formatting root is kept
        self.assertNotIn("5", ids)      # the section above it is capped away


@unittest.skipUnless(_has_graph_deps(), "regulatory_testgen.graph deps not installed")
class TestGenerationPromptBuilder(unittest.TestCase):
    def test_prompt_embeds_parent_and_referenced_text(self):
        import generator
        graph = _build_sample_graph()
        base = "The vehicle shall detect a stationary target ahead."
        enriched = generator.build_generation_context_text(graph, "5.2.1", base)
        self.assertIn(base, enriched)
        self.assertIn("RELATED REGULATORY CONTEXT", enriched)
        # parent section 5.2 and referenced clause 6.1 text present
        self.assertIn("Requirements for car-to-car scenarios.", enriched)
        self.assertIn("A dry paved road surface is required.", enriched)
        # unrelated section text absent
        self.assertNotIn("Completely unrelated content here.", enriched)

    def test_no_graph_returns_base_text_unchanged(self):
        import generator
        self.assertEqual(generator.build_generation_context_text(None, "5.2.1", "base"), "base")


# ── retrieval.load_clause_index shape ────────────────────────────────────────

class TestClauseIndex(unittest.TestCase):
    def test_index_shape_if_checkpoints_present(self):
        import retrieval
        if not (retrieval.CHECKPOINTS / "01_clauses.json").exists():
            self.skipTest("clause checkpoints not present in this checkout")
        index = retrieval.load_clause_index()
        self.assertGreater(len(index), 0)
        sample = next(iter(index.values()))
        self.assertEqual(
            set(sample),
            {"title", "text", "text_preview", "category", "force", "testable",
             "section_path", "references", "parent"},
        )
        self.assertLessEqual(len(sample["text_preview"]), 400)
        # Annex namespacing: body §5 and Annex 3's §5 are distinct, and body §5
        # is "Specifications" (not the annex's "Reporting by Technical Service").
        self.assertEqual(index["5"]["title"], "Specifications")
        self.assertIsNone(index["5"]["parent"])
        if "Annex 3 / 5" in index:
            self.assertEqual(index["Annex 3 / 5"]["parent"], "Annex 3")


# ── Document repair / normalization (normalize.py) ───────────────────────────

def _raw(cid, title="", text="", region="specifications", path=None, pseudo=False, refs=None):
    """Build a raw parsed-clause dict like 01_clauses.json entries."""
    return {
        "clause_id": cid,
        "title": title,
        "text": text,
        "document_region": region,
        "section_path": path if path is not None else [],
        "is_pseudo_clause": pseudo,
        "references": refs or [],
    }


class TestNormalize(unittest.TestCase):
    def test_natkey_orders_numerically(self):
        from normalize import natkey
        # 5.10 sorts after 5.2 (not lexically), and section 10 after section 5.
        self.assertEqual(sorted(["5.10", "5.2", "5.1"], key=natkey), ["5.1", "5.2", "5.10"])
        self.assertEqual(sorted(["10", "2", "5"], key=natkey), ["2", "5", "10"])

    def test_formfield_and_junk_detection(self):
        from normalize import looks_like_formfield, is_form_junk
        self.assertTrue(looks_like_formfield("Trademark:.."))
        self.assertTrue(looks_like_formfield("Type: ..... ......................"))
        self.assertFalse(looks_like_formfield("Specific Requirements"))
        # junk via Communication section_path
        self.assertTrue(is_form_junk(_raw("5", title="", path=["Communication"])))
        # junk via form-field text with a dotted fill-in
        self.assertTrue(is_form_junk(_raw("1.3", text="Means of identification: ...............")))
        self.assertFalse(is_form_junk(_raw("5.2", title="Specific Requirements")))
        # A full sentence ending in a colon (introducing a list/table) is NOT a
        # form field — regression for 5.2.1.4 "…as shown in the following table:"
        # being pruned along with its performance table.
        sentence = ("In absence of driver's input which would lead to interruption "
                    "according to paragraph 5.3.2., the AEBS shall be able to achieve "
                    "a relative impact speed less or equal to the following table:")
        self.assertFalse(looks_like_formfield(sentence))
        self.assertFalse(is_form_junk(_raw("5.2.1.4", title="Speed reduction by braking demand",
                                            text=sentence)))

    def test_prune_keeps_clause_whose_text_introduces_a_table(self):
        """A performance clause whose body ends in a colon before its table must
        survive pruning (regression: 5.2.1.4 was silently dropped)."""
        from normalize import prune_form_fields
        best = {
            "5.2.1.4": _raw("5.2.1.4", title="Speed reduction by braking demand",
                            text="The AEBS shall achieve a speed as shown in the following table:"),
        }
        self.assertIn("5.2.1.4", prune_form_fields(best))

    def test_choose_best_copy_prefers_body_over_form(self):
        from normalize import choose_best_copies
        copies = [
            _raw("5", title="Brief description of vehicle:.", path=["Communication"]),
            _raw("5", title="", text="Reporting by Technical Service " * 5, path=[]),
        ]
        best = choose_best_copies(copies)
        # the Communication form-field copy must NOT win
        self.assertNotEqual(best["5"]["title"], "Brief description of vehicle:.")

    def test_prune_drops_leaf_forms_keeps_polluted_parents(self):
        from normalize import prune_form_fields
        best = {
            "1": _raw("1", title="Scope", text="This Regulation applies…"),
            "1.1": _raw("1.1", title="Vehicle make: ."),          # leaf form → drop
            "6": _raw("6", title="Date of submission of vehicle for approval:"),  # parent → keep, blanked
            "6.1": _raw("6.1", title="Test Conditions", path=["Test procedure"]),
        }
        pruned = prune_form_fields(best)
        self.assertIn("1", pruned)
        self.assertNotIn("1.1", pruned)      # leaf form field dropped
        self.assertIn("6", pruned)           # section parent kept
        self.assertEqual(pruned["6"]["title"], "")  # blanked for title recovery

    def test_repair_recovers_titles(self):
        from normalize import repair_titles
        index = {
            "5":     {"title": "", "text": "", "section_path": []},
            "5.1":   {"title": "", "text": "General requirements", "section_path": ["Specifications"]},
            "5.2":   {"title": "Specific Requirements", "text": "", "section_path": ["Specifications"]},
            "5.2.1": {"title": "", "text": "Car to car scenario", "section_path": ["Specifications"]},
        }
        repair_titles(index)
        # top-level section recovered from children's shared section_path
        self.assertEqual(index["5"]["title"], "Specifications")
        # heading-in-text recovered
        self.assertEqual(index["5.1"]["title"], "General requirements")
        self.assertEqual(index["5.2.1"]["title"], "Car to car scenario")
        # existing good title preserved
        self.assertEqual(index["5.2"]["title"], "Specific Requirements")

    def test_repair_ignores_bare_number_lines_and_strips_toc(self):
        from normalize import repair_titles
        index = {
            "5.1.1": {"title": "", "text": "5.1.1\nAny vehicle fitted with an AEBS shall…",
                      "section_path": ["Specifications"]},
            "3": {"title": "Application for approval . 6", "text": "", "section_path": []},
        }
        repair_titles(index)
        self.assertNotEqual(index["5.1.1"]["title"], "5.1.1")   # bare number skipped
        self.assertEqual(index["3"]["title"], "Application for approval")  # TOC page num stripped


class TestStructureAndClauseTools(unittest.TestCase):
    """The rewritten get_document_structure / get_clause using the repaired index."""

    def _index(self):
        return {
            "5":     {"title": "Specifications", "text": "", "category": "informative",
                      "section_path": [], "references": [], "parent": None},
            "5.1":   {"title": "General requirements", "text": "General requirements",
                      "category": "obligation", "section_path": ["Specifications"], "references": [], "parent": "5"},
            "5.2":   {"title": "Specific Requirements", "text": "", "category": "informative",
                      "section_path": ["Specifications"], "references": [], "parent": "5"},
            "5.2.1": {"title": "Car to car scenario", "text": "Car to car scenario",
                      "category": "informative", "section_path": ["Specifications"], "references": ["6.1"], "parent": "5.2"},
            "5.2.1.1": {"title": "Collision warning", "text": "The system shall warn…",
                        "category": "obligation", "section_path": ["Specifications"], "references": [], "parent": "5.2.1"},
            "6.1":   {"title": "Test Conditions", "text": "Dry road…", "category": "test_condition",
                      "section_path": ["Test procedure"], "references": [], "parent": None},
        }

    def test_structure_overview_lists_sections_and_children(self):
        from tools import handle_get_document_structure
        out = handle_get_document_structure(self._index())
        self.assertIn("**5** Specifications", out)
        self.assertIn("5.1", out)
        self.assertIn("5.2", out)

    def test_structure_section_drill_shows_nested_scenarios(self):
        from tools import handle_get_document_structure
        out = handle_get_document_structure(self._index(), section="5")
        # the scenario level (depth 3) must be visible when drilling into 5
        self.assertIn("5.2.1", out)
        self.assertIn("Car to car scenario", out)
        self.assertIn("5.2.1.1", out)

    def test_get_clause_shows_full_text_parent_children_refs(self):
        from tools import handle_get_clause
        out = handle_get_clause("5.2.1", self._index())
        self.assertIn("Car to car scenario", out)
        self.assertIn("Parent: 5.2", out)              # parent section
        self.assertIn("5.2.1.1", out)                  # direct sub-clause listed
        self.assertIn("6.1", out)                       # cross-reference listed


class TestAnnexNamespacing(unittest.TestCase):
    def test_parent_of_body_and_annex(self):
        from normalize import parent_of
        self.assertIsNone(parent_of("5"))
        self.assertEqual(parent_of("5.2.1"), "5.2")
        self.assertEqual(parent_of("Annex 3 / 5"), "Annex 3")
        self.assertEqual(parent_of("Annex 3 / 5.1"), "Annex 3 / 5")
        self.assertEqual(parent_of("Annex 3 - Appendix 1"), "Annex 3")
        self.assertIsNone(parent_of("Annex 3"))

    def test_assign_structure_namespaces_annex_by_line_order(self):
        from normalize import assign_structure
        raw = [
            _raw("5", text="Specifications body"),                      # body §5
            {"clause_id": "Annex 3", "text": "Special requirements",
             "document_region": "annex", "line_start": 100, "section_path": [], "is_pseudo_clause": False},
            _raw("5", text="Reporting by Technical Service"),           # annex §5 (mis-tagged)
        ]
        # give line numbers: body before annex heading, annex clause after it
        raw[0]["line_start"] = 10
        raw[2]["line_start"] = 110
        out = assign_structure(raw)
        ids = [c["clause_id"] for c in out]
        self.assertIn("5", ids)              # body §5 keeps bare id
        self.assertIn("Annex 3 / 5", ids)    # annex §5 namespaced
        self.assertIn("Annex 3", ids)

    def test_body_and_annex_five_do_not_collide(self):
        from normalize import assign_structure, choose_best_copies
        raw = [
            dict(_raw("5", text="Specifications heading here"), line_start=10),
            dict(clause_id="Annex 3", text="Special requirements", document_region="annex",
                 line_start=100, section_path=[], is_pseudo_clause=False, title="", references=[]),
            dict(_raw("5", text="Reporting by Technical Service"), line_start=110),
        ]
        best = choose_best_copies(assign_structure(raw))
        self.assertIn("5", best)
        self.assertIn("Annex 3 / 5", best)
        self.assertNotEqual(best["5"], best["Annex 3 / 5"])


class TestGraphViz(unittest.TestCase):
    def test_category_colors_are_mutually_distinguishable(self):
        """Guard against reintroducing confusable category colours (e.g. the old
        performance_data mauve vs unknown violet). All substantive categories
        must be well separated; neutral meta-labels (unknown/table/unmatched/
        _default) are exempt — 'unknown' is deliberately a grey catch-all so it
        never competes with a real category's hue."""
        from graph_viz import CATEGORY_COLOR
        import pdf_highlight

        def h2rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

        def dist(a, b):
            r1, g1, b1 = a; r2, g2, b2 = b
            rm = (r1 + r2) / 2
            dr, dg, db = r1 - r2, g1 - g2, b1 - b2
            return ((2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db) ** 0.5

        _META = {"unknown"}   # neutral grey catch-all — exempt from hue separation
        cats = [(n, h) for n, h in CATEGORY_COLOR.items() if n not in _META]
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                (na, ha), (nb, hb) = cats[i], cats[j]
                d = dist(h2rgb(ha), h2rgb(hb))
                self.assertGreaterEqual(
                    d, 110, f"{na} ({ha}) and {nb} ({hb}) are too similar (dist={d:.0f})")

        # The pdf_highlight fallback must stay in sync with the graph palette.
        for cat, hexc in CATEGORY_COLOR.items():
            want = tuple(x / 255 for x in h2rgb(hexc))
            got = pdf_highlight.DEFAULT_CATEGORY_RGB.get(cat)
            self.assertIsNotNone(got, f"{cat} missing from DEFAULT_CATEGORY_RGB")
            self.assertTrue(all(abs(a - b) < 0.01 for a, b in zip(want, got)),
                            f"{cat}: graph {want} != pdf_highlight {got}")

    def test_edges_from_index_uses_parent_and_references(self):
        from graph_viz import edges_from_index
        index = {
            "5": {"parent": None, "references": []},
            "5.2": {"parent": "5", "references": []},
            "5.2.1": {"parent": "5.2", "references": ["6.1"]},
            "6.1": {"parent": None, "references": []},
        }
        edges = set(edges_from_index(index))
        self.assertIn(("5", "CONTAINS", "5.2"), edges)
        self.assertIn(("5.2", "CONTAINS", "5.2.1"), edges)
        self.assertIn(("5.2.1", "REFERS_TO", "6.1"), edges)


    EDGES = [
        ("5", "CONTAINS", "5.2"),
        ("5.2", "CONTAINS", "5.2.1"),
        ("5.2.1", "REFERS_TO", "6.1"),
        ("6", "CONTAINS", "6.1"),
    ]

    def test_neighborhood_one_hop_both_edge_types(self):
        from graph_viz import neighborhood
        nodes, sub = neighborhood(self.EDGES, "5.2.1", hops=1)
        self.assertEqual(nodes, {"5.2.1", "5.2", "6.1"})   # parent + cross-ref, not '5' (2 hops)
        self.assertIn(("5.2", "CONTAINS", "5.2.1"), sub)
        self.assertIn(("5.2.1", "REFERS_TO", "6.1"), sub)
        self.assertNotIn(("5", "CONTAINS", "5.2"), sub)     # endpoint '5' out of range

    def test_neighborhood_respects_edge_type_filter(self):
        from graph_viz import neighborhood
        nodes, sub = neighborhood(self.EDGES, "5.2.1", hops=1, edge_types=("REFERS_TO",))
        self.assertEqual(nodes, {"5.2.1", "6.1"})           # CONTAINS parent not traversed
        self.assertEqual(sub, [("5.2.1", "REFERS_TO", "6.1")])

    def test_neighborhood_two_hops_reaches_grandparent(self):
        from graph_viz import neighborhood
        nodes, _sub = neighborhood(self.EDGES, "5.2.1", hops=2)
        self.assertIn("5", nodes)                            # grandparent reached at 2 hops

    def test_filter_edges_full_graph(self):
        from graph_viz import filter_edges
        nodes, kept = filter_edges(self.EDGES, ("CONTAINS", "REFERS_TO"))
        self.assertEqual(nodes, {"5", "5.2", "5.2.1", "6", "6.1"})
        self.assertEqual(len(kept), 4)
        nodes_c, kept_c = filter_edges(self.EDGES, ("REFERS_TO",))
        self.assertEqual(kept_c, [("5.2.1", "REFERS_TO", "6.1")])


class TestClassifyHelpers(unittest.TestCase):
    def test_is_testable_respects_force(self):
        from regulatory_testgen.classify import is_testable
        self.assertTrue(is_testable({"category": "obligation", "force": "binding"}))
        self.assertFalse(is_testable({"category": "obligation", "force": "example"}))
        self.assertFalse(is_testable({"category": "definition", "force": "binding"}))
        self.assertTrue(is_testable("obligation"))        # legacy string
        self.assertFalse(is_testable("informative"))
        self.assertFalse(is_testable(None))

    def test_normalize_classifications_legacy_and_rich(self):
        from regulatory_testgen.classify import normalize_classifications
        out = normalize_classifications({
            "uid-a": "obligation",                                  # legacy
            "uid-b": {"category": "informative", "force": "example"},
        })
        self.assertEqual(out["uid-a"], {"category": "obligation", "force": "binding", "reasoning": ""})
        self.assertEqual(out["uid-b"]["force"], "example")

    def test_parent_key_body_and_annex(self):
        from regulatory_testgen.classify import _parent_key
        self.assertIsNone(_parent_key("5"))
        self.assertEqual(_parent_key("5.2.1"), "5.2")
        self.assertEqual(_parent_key("Annex 3 / 5"), "Annex 3")
        self.assertEqual(_parent_key("Annex 3 - Appendix 1"), "Annex 3")
        self.assertIsNone(_parent_key("Annex 3"))

    def test_assign_keys_namespaces_annex_by_line_order(self):
        from regulatory_testgen.classify import _assign_keys

        class C:
            def __init__(self, cid, line, region="", pseudo=False):
                self.clause_id = cid
                self.line_start = line
                self.document_region = region
                self.is_pseudo_clause = pseudo
                self.title = ""
                self.text = ""
        clauses = [C("5", 10), C("Annex 3", 100, "annex"), C("5", 110)]
        keys = [k for k, _c in _assign_keys(clauses)]
        self.assertIn("5", keys)
        self.assertIn("Annex 3", keys)
        self.assertIn("Annex 3 / 5", keys)

    def test_build_clause_index_resolves_categories_by_uid(self):
        """Regression: uploads keyed classifications by uid; the index must carry
        `uid` so lookups resolve. Without it every clause defaults to 'unknown'."""
        from retrieval import build_clause_index
        raw = [
            {"clause_id": "5.2.1", "uid": "clause-5-2-1-abc", "title": "Scenario",
             "text": "The system shall brake.", "section_path": [], "references": [],
             "is_pseudo_clause": False, "line_start": 10},
        ]
        classifications = {"clause-5-2-1-abc": {"category": "obligation", "force": "binding"}}
        idx = build_clause_index(raw, classifications)
        self.assertEqual(idx["5.2.1"]["category"], "obligation")
        self.assertTrue(idx["5.2.1"]["testable"])
        # Missing uid -> unknown (documents the failure the fix prevents).
        raw_no_uid = [{k: v for k, v in raw[0].items() if k != "uid"}]
        idx2 = build_clause_index(raw_no_uid, classifications)
        self.assertEqual(idx2["5.2.1"]["category"], "unknown")

    def test_ingest_raw_dict_includes_uid(self):
        """Guard the exact shape ingest._build_clause_index feeds the index."""
        import ingest

        class C:
            clause_id = "5.2.1"
            uid = "clause-5-2-1-abc"
            title = "Scenario"
            text = "The system shall brake."
            document_region = ""
            section_path: list = []
            references: list = []
            is_pseudo_clause = False
            line_start = 10
        classifications = {"clause-5-2-1-abc": {"category": "obligation", "force": "binding"}}
        idx = ingest._build_clause_index([C()], classifications)
        self.assertEqual(idx["5.2.1"]["category"], "obligation")


class TestPdfCategoryOverlay(unittest.TestCase):
    def test_hex_to_rgb01(self):
        from pdf_highlight import hex_to_rgb01
        self.assertEqual(hex_to_rgb01("#ff0000"), (1.0, 0.0, 0.0))
        r, g, b = hex_to_rgb01("#e06666")
        self.assertAlmostEqual(r, 224 / 255)

    def test_category_boxes_by_page_groups_and_filters(self):
        from pdf_highlight import category_boxes_by_page
        cpm = {
            "5.2.1": [{"page": 3, "bbox": [0, 0, 1, 1]}],
            "6.1": [{"page": 3, "bbox": [1, 1, 2, 2]}, {"page": 4, "bbox": [0, 0, 1, 1]}],
        }
        cat = {"5.2.1": "obligation", "6.1": "test_condition"}
        by = category_boxes_by_page(cpm, cat)
        self.assertEqual(len(by[3]), 2)
        self.assertEqual(by[4], [([0, 0, 1, 1], "test_condition")])
        only = category_boxes_by_page(cpm, cat, only_categories={"obligation"})
        self.assertEqual(only, {3: [([0, 0, 1, 1], "obligation")]})

    def test_extraction_overlay_colors_matched_and_tables_only(self):
        import json
        import os
        import tempfile
        from pdf_highlight import all_block_regions, build_extraction_overlay
        mid = {"pdf_info": [{"page_idx": 0, "para_blocks": [
            {"type": "text", "bbox": [0, 0, 10, 10],
             "lines": [{"spans": [{"type": "text", "content": "hello"}]}], "index": 0},
            {"type": "text", "bbox": [0, 12, 10, 22],
             "lines": [{"spans": [{"type": "text", "content": "unmatched text"}]}], "index": 1},
            {"type": "table", "bbox": [0, 30, 50, 80], "blocks": [], "index": 2},
            {"type": "image", "bbox": [0, 90, 10, 99], "index": 3},
        ]}]}
        path = os.path.join(tempfile.mkdtemp(), "m.json")
        with open(path, "w") as f:
            f.write(json.dumps(mid))

        regions = all_block_regions(Path(path))
        self.assertEqual(len(regions), 3)   # image skipped; text+text+table kept

        cpm = {"5": [{"page": 0, "bbox": [0, 0, 10, 10]}]}   # only the first text block matched a clause
        overlay = build_extraction_overlay(cpm, {"5": "obligation"}, Path(path))
        cats = sorted(c for (_b, c) in overlay[0])
        # The second text block matched no clause and is now left unhighlighted
        # (no "unmatched" region); tables and clause-matched blocks still show.
        self.assertEqual(cats, ["obligation", "table"])

    def test_render_category_page_images_smoke(self):
        try:
            import fitz  # noqa: F401
        except Exception:
            self.skipTest("PyMuPDF not installed")
        import os
        import tempfile
        from pdf_highlight import render_category_page_images
        doc = fitz.open()
        doc.new_page(width=200, height=200)
        path = os.path.join(tempfile.mkdtemp(), "t.pdf")
        doc.save(path)
        doc.close()
        cpm = {"5": [{"page": 0, "bbox": [10, 10, 100, 30]}]}
        imgs = render_category_page_images(
            Path(path), cpm, {"5": "obligation"}, page_indices=[0], zoom=1.0,
        )
        self.assertEqual(len(imgs), 1)
        self.assertEqual(imgs[0][0], 1)                       # 1-indexed page number
        self.assertTrue(imgs[0][1].startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
