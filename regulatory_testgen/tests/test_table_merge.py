"""Tests for merge_multipage_tables — stitching MinerU's page-split tables.

Run from the thesis-code root, ideally in the pipeline venv:

    python -m unittest regulatory_testgen.tests.test_table_merge -v

The cases mirror what the real evaluation corpus contains: a genuine
page-spanning table that repeats its header (R157's country list, MERGE), and
the several look-alikes that must stay separate — different column counts
(R48's chromaticity tables) and same-column adjacent tables with no repeated
header (R48's coordinate tables on one page).
"""
from __future__ import annotations

import re
import unittest

from regulatory_testgen.parsing.table_merge import merge_multipage_tables


def _ntables(s: str) -> int:
    return len(re.findall(r"<table\b", s, re.IGNORECASE))


def _nrows(table_html: str) -> int:
    return len(re.findall(r"<tr\b", table_html, re.IGNORECASE))


HDR = "<tr><td>Country</td><td>Assessed</td><td>Comments</td></tr>"


def _table(header: str, *body_rows: str) -> str:
    return "<table>" + header + "".join(body_rows) + "</table>"


class MergeMultipageTables(unittest.TestCase):
    def test_repeated_header_across_page_merges(self):
        a = _table(HDR, "<tr><td>E 28 Belarus</td><td></td><td></td></tr>")
        b = _table(HDR, "<tr><td>E 29 Estonia</td><td></td><td></td></tr>")
        md = f"{a}\n\nE/ECE/TRANS/505/Rev.3/Add.156 Annex 1 - Appendix\n{b}"
        merged, n = merge_multipage_tables(md)
        self.assertEqual(n, 1)
        self.assertEqual(_ntables(merged), 1)
        self.assertIn("Belarus", merged)
        self.assertIn("Estonia", merged)
        # header kept exactly once; furniture line dropped
        self.assertEqual(merged.count("Assessed"), 1)
        self.assertNotIn("TRANS/505", merged)
        self.assertEqual(_nrows(merged), 3)  # header + 2 country rows

    def test_different_column_count_not_merged(self):
        a = _table("<tr><td>A</td><td>B</td><td>C</td></tr>", "<tr><td>1</td><td>2</td><td>3</td></tr>")
        b = _table("<tr><td>X</td><td>Y</td></tr>", "<tr><td>4</td><td>5</td></tr>")
        merged, n = merge_multipage_tables(f"{a}\n\n{b}")
        self.assertEqual(n, 0)
        self.assertEqual(_ntables(merged), 2)

    def test_same_columns_but_no_repeated_header_not_merged(self):
        # R48 210->211: two 3-col tables, adjacent, but different headers/content.
        a = _table("<tr><td>X</td><td></td><td>y</td></tr>", "<tr><td>W1</td><td>0.310</td><td>0.348</td></tr>")
        b = _table("<tr><td>SY12</td><td>green</td><td>y=1.29x</td></tr>", "<tr><td>SY23</td><td>locus</td><td></td></tr>")
        merged, n = merge_multipage_tables(f"{a}\n\n{b}")
        self.assertEqual(n, 0)
        self.assertEqual(_ntables(merged), 2)

    def test_real_content_between_blocks_not_merged(self):
        # A markdown heading between two identical-header tables = a real boundary.
        a = _table(HDR, "<tr><td>E 1</td><td></td><td></td></tr>")
        b = _table(HDR, "<tr><td>E 2</td><td></td><td></td></tr>")
        md = f"{a}\n\n# 5. A new section with a genuine paragraph of prose here.\n\n{b}"
        merged, n = merge_multipage_tables(md)
        self.assertEqual(n, 0)
        self.assertEqual(_ntables(merged), 2)

    def test_clause_number_between_blocks_not_merged(self):
        a = _table(HDR, "<tr><td>E 1</td><td></td><td></td></tr>")
        b = _table(HDR, "<tr><td>E 2</td><td></td><td></td></tr>")
        md = f"{a}\n\n5.2.1 The system shall do a thing distinct from the table.\n{b}"
        merged, n = merge_multipage_tables(md)
        self.assertEqual(n, 0)
        self.assertEqual(_ntables(merged), 2)

    def test_three_page_chain_merges_twice(self):
        a = _table(HDR, "<tr><td>E 1</td><td></td><td></td></tr>")
        b = _table(HDR, "<tr><td>E 2</td><td></td><td></td></tr>")
        c = _table(HDR, "<tr><td>E 3</td><td></td><td></td></tr>")
        md = f"{a}\n1\n{b}\n2\n{c}"
        merged, n = merge_multipage_tables(md)
        self.assertEqual(n, 2)
        self.assertEqual(_ntables(merged), 1)
        self.assertEqual(_nrows(merged), 4)  # header + 3 rows

    def test_cell_html_preserved(self):
        math = r"$\mathbf { A } _ { 1 2 }$"
        a = _table(HDR, f"<tr><td>{math}</td><td>green</td><td>y=1</td></tr>")
        b = _table(HDR, "<tr><td>E 29 Estonia</td><td></td><td></td></tr>")
        merged, n = merge_multipage_tables(f"{a}\n\n{b}")
        self.assertEqual(n, 1)
        self.assertIn(math, merged)  # LaTeX cell content untouched

    def test_case_sensitive_header_not_merged(self):
        # Near-duplicate per-figure tables (R48 1631/1635) differ only in case.
        a = _table("<tr><td></td><td>Illuminating surface</td><td>Declared light-emitting</td></tr>",
                   "<tr><td>Edges</td><td>a and b</td><td>c and d</td></tr>")
        b = _table("<tr><td></td><td>Illuminating surface</td><td>Declared Light-emitting</td></tr>",
                   "<tr><td>Edges</td><td>aand b</td><td>c and d</td></tr>")
        merged, n = merge_multipage_tables(f"{a}\n\n{b}")
        self.assertEqual(n, 0)
        self.assertEqual(_ntables(merged), 2)

    def test_no_tables_is_noop(self):
        md = "# Heading\n\nJust some prose, no tables at all.\n"
        self.assertEqual(merge_multipage_tables(md), (md, 0))


if __name__ == "__main__":
    unittest.main()
