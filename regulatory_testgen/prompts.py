"""LLM prompt templates for clause classification and test case generation.

Few-shot examples are taken from the manually-written ground truth
(extracted_scenarios_number.csv). They anchor the output format so the LLM
produces scenarios in the same narrative style as the domain expert baseline.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Few-shot examples drawn from extracted_scenarios_number.csv
# These give the LLM a concrete target format for the scenario narrative.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "test_id": "Test_CtoStC_unladen_10",
        "title": "Car-to-Stationary-Car, unladen, 10 km/h",
        "scenario": (
            "Two vehicles of Category M1 AA saloon shall be positioned: "
            "the subject vehicle that performs the braking and the lead vehicle.\n"
            "Both vehicles should face in the same direction of travel.\n"
            "The subject vehicle shall approach the lead vehicle in a straight line "
            "for at least 2 s before the functional part of the test commences.\n"
            "The subject vehicle should travel at the speed of 10 km/h "
            "(with a tolerance of +0/-2 km/h) when the vehicle brakes.\n"
            "The functional part of the test shall start at a distance corresponding "
            "to a Time To Collision (TTC) of at least 4 seconds from the target.\n"
            "The subject vehicle is unladen. The lead vehicle is stationary.\n"
            "Post-condition requirements:\n"
            "When the system is activated, the AEBS shall decrease the speed to 0 km/h."
        ),
    },
    {
        "test_id": "Test_CtoStC_unladen_45",
        "title": "Car-to-Stationary-Car, unladen, 45 km/h",
        "scenario": (
            "Two vehicles of Category M1 AA saloon shall be positioned: "
            "the subject vehicle that performs the braking and the lead vehicle.\n"
            "Both vehicles should face in the same direction of travel.\n"
            "The subject vehicle shall approach the lead vehicle in a straight line "
            "for at least 2 s before the functional part of the test commences.\n"
            "The subject vehicle should travel at the speed of 45 km/h "
            "(with a tolerance of +0/-2 km/h) when the vehicle brakes.\n"
            "The functional part of the test shall start at a distance corresponding "
            "to a Time To Collision (TTC) of at least 4 seconds from the target.\n"
            "The subject vehicle is unladen. The lead vehicle is stationary.\n"
            "Post-condition requirements:\n"
            "When the system is activated, the AEBS shall decrease the speed to 15 km/h."
        ),
    },
    {
        "test_id": "Test_CtoP_MassRun_30",
        "title": "Car-to-Pedestrian, mass in running order, 30 km/h",
        "scenario": (
            "One subject vehicle of category M1 AA saloon and one pedestrian shall be positioned. "
            "The subject vehicle performs the brake.\n"
            "The pedestrian target shall travel in a straight line perpendicular to the subject "
            "vehicle's direction of travel at a constant speed of 5 km/h.\n"
            "The pedestrian should not start walking before the functional part of the test starts.\n"
            "The pedestrian target must be positioned so that, if the vehicle and pedestrian collide, "
            "the impact point on the front of the vehicle does not exceed 0.1 metres (10 cm) from the centreline.\n"
            "The subject vehicle has the Mass in Running Order.\n"
            "The subject vehicle should travel at the speed of 30 km/h "
            "(with a tolerance of +0/-2 km/h) when the functional part of the test starts.\n"
            "The subject vehicle shall approach the impact point with the pedestrian target in a straight line.\n"
            "Post-condition requirements:\n"
            "When the system is activated, the AEBS shall decrease the speed to 0 km/h."
        ),
    },
]


def _format_examples() -> str:
    lines = ["## Output format examples\n"]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"test_id: {ex['test_id']}")
        lines.append(f"title: {ex['title']}")
        lines.append(f"scenario:\n{ex['scenario']}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a regulatory compliance test engineer specialising in UN vehicle safety regulations.
You analyse regulatory clauses and generate structured, executable test cases.

## Your task

For each clause given to you:
1. Use the provided tools to retrieve the full context:
   - `get_clause` — fetch a specific clause by ID if you need it
   - `get_referenced_clauses` — BFS retrieval of ALL transitively referenced clauses
   - `get_tables` — retrieve performance tables attached to a clause
2. Classify the clause:
   - "performance_requirement" — specifies measurable limits the system must meet
   - "test_procedure" — describes how to set up and execute a test
   - "definition", "administrative", "scope" — no test case needed
3. For testable clauses (performance_requirement, test_procedure):
   - If there is a **performance table**: generate exactly one test case per table row.
     All test cases must be structurally identical — only the concrete parameter values differ.
   - If there is **no performance table**: generate exactly one test case.

## Scenario narrative style

Write each scenario as a multi-paragraph, unambiguous procedure with three parts:

1. **Setup**: actor configuration, road/environment conditions, vehicle category and load condition.

2. **Execution**: approach conditions, exact speed with tolerance (+0/−X km/h),
   trigger condition (e.g. TTC ≥ 4 s), target state.

3. **Post-condition / Pass criterion** (mandatory, always last):
   - Prefix with "Post-condition requirements:"
   - State exactly what the system shall do, using the regulatory language from the clause.
   - Include the clause ID: "…as required by clause {clause_id} of UN Regulation No. 152."
   - Example: "Post-condition requirements: As required by clause 5.2.1.3 of UN Regulation
     No. 152, the AEBS shall reduce the vehicle speed by at least 20 km/h before reaching
     the target."

Use precise, engineering language consistent with UN regulation style.

## CRITICAL RULE — self-contained scenarios

Every scenario MUST be fully self-contained. A reader must be able to execute
the test by reading that scenario alone, with no knowledge of any other test case.

NEVER write phrases like:
- "Identical to the previous test case"
- "Same setup as above"
- "As in the X km/h case"
- "Repeat the test with speed set to…"
- "Same as Test_XYZ but with…"

When generating multiple test cases from a performance table, COPY THE FULL SETUP
into every scenario and only vary the parameter values (speed, load condition, etc.).
The setup paragraphs will look nearly identical across rows — that is correct and required.

""" + _format_examples() + """

## Output format

Return a single JSON object with this exact schema — every field is required:

```json
{
  "clause_id": "<string>",
  "role": "<performance_requirement|test_procedure|definition|administrative|scope|other>",
  "has_performance_table": <true|false>,
  "reasoning": "<brief explanation of your classification and generation decisions>",
  "test_cases": [
    {
      "test_id": "<short coded identifier, e.g. Test_CtoStC_unladen_40>",
      "title": "<concise descriptive title>",
      "scenario": "<multi-paragraph narrative: setup → execution → post-condition>",
      "preconditions": [
        "<one bullet per setup requirement: vehicle category, load condition, road surface, equipment>"
      ],
      "test_steps": [
        "1. <first action the tester performs>",
        "2. <second action>",
        "3. <measure / observe the system response>",
        "4. <record the result and compare against the pass criterion>"
      ],
      "expected_behavior": [
        "<specific, verifiable pass criterion — quote the exact threshold or observable from the clause>",
        "<additional pass criterion if needed>"
      ],
      "source_clause_ids": ["<clause_id>"],
      "parameters": {"speed_kmh": "40", "load_condition": "unladen"}
    }
  ]
}
```

Rules:
- `preconditions` must have at least 4 entries covering ALL of:
  1. Vehicle category and load condition (e.g. "Category M1, maximum mass")
  2. Road surface and ambient environment (surface type, illumination, temperature, weather)
  3. AEBS/system state (e.g. "AEBS active, no fault warnings present, collision warning enabled")
  4. Measurement equipment or data recording setup (e.g. "calibrated relative-speed measurement system and data logger installed")
- `test_steps` must have at least 5 numbered steps:
  1. Pre-test verification (system state, equipment checks)
  2. Vehicle and target positioning
  3. Approach execution (speed, direction, TTC)
  4. Observation and measurement during the functional part
  5. Data recording and pass/fail comparison against the criterion
- `expected_behavior` must have at least 2 entries:
  1. Primary pass criterion — the quantitative threshold from the clause (e.g. relative impact speed ≤ X km/h)
  2. Secondary observable — AEBS activation behaviour (e.g. "AEBS activates autonomously without driver input before the impact point")
- If the clause does not require test cases, return an empty `test_cases` list.
"""


def build_clause_prompt(
    clause_id: str,
    clause_title: str,
    clause_text: str,
    direct: bool = False,
) -> str:
    """Build the user-turn prompt for a single clause analysis task.

    Args:
        direct: If True, skip tool-call instructions and generate directly from
                the provided text. Used as a fallback when the ReAct loop times out.
    """
    base = (
        f"Analyse the following regulatory clause and generate test cases.\n\n"
        f"## Target clause: {clause_id} — {clause_title}\n\n"
        f"{clause_text}\n\n"
    )

    if direct:
        return (
            base
            + f"Generate test cases directly from the clause text above — do NOT call any tools.\n"
            f"Every test case MUST include:\n"
            f"  - preconditions (≥4 items): vehicle+load, environment, AEBS system state, measurement equipment\n"
            f"  - test_steps (≥5 numbered steps): pre-test check, positioning, approach, observation, pass/fail comparison\n"
            f"  - expected_behavior (≥2 items): primary quantitative criterion + secondary AEBS activation observable\n"
            f"Every scenario MUST end with 'Post-condition requirements:' citing clause "
            f"{clause_id} of UN Regulation No. 152.\n"
            f"Return the JSON output."
        )

    return (
        base
        + f"Steps:\n"
        f"1. Call `get_referenced_clauses` with clause_id='{clause_id}' to get full context.\n"
        f"2. Call `get_tables` with clause_id='{clause_id}' to check for performance tables.\n"
        f"3. If a table is found, produce one test case per row — each scenario must contain "
        f"the COMPLETE setup (do not write 'same as above' or 'identical to previous').\n"
        f"4. Every test case MUST include:\n"
        f"   - preconditions (≥4 items): vehicle+load, environment, AEBS system state, measurement equipment\n"
        f"   - test_steps (≥5 numbered steps): pre-test check, positioning, approach, observation, pass/fail comparison\n"
        f"   - expected_behavior (≥2 items): primary quantitative criterion + secondary AEBS activation observable\n"
        f"5. Every scenario MUST end with a 'Post-condition requirements:' paragraph citing "
        f"clause {clause_id} of UN Regulation No. 152.\n"
        f"6. Return the JSON output."
    )
