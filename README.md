# Schema Field Mapper — LangGraph Pipeline

An autonomous AI pipeline that maps every field from a **MySQL source schema** to a **MongoDB destination schema** using a multi-step LangGraph workflow with LangChain tools, confidence scoring, and human-in-the-loop review.

The pipeline breaks the work into multiple autonomous steps ingesting each schema independently, discovering structure, asking clarifying questions, then mapping table-by-table and field-by-field.

---

## Tech Stack

| Technology | Role |
|---|---|
| **LangGraph** | Cyclic state machine for agentic workflow orchestration |
| **LangChain** | Prompt templates, tool calling, chain composition |
| **OpenAI GPT-4o** | LLM for schema analysis, semantic mapping, and discovery |
| **Python** | Core runtime |

---

## System Workflow

!([Workflow_Dhanya_Assignment.png](https://github.com/dhanyavasantha/SchemaFieldMapper/blob/main/Workflow_Dhanya_Assignment.png?raw=true))

### Pipeline Steps

| Step | Node | Description |
|---|---|---|
| 1 | `ingest_source` | Load and parse the MySQL source schema JSON |
| 2 | `ingest_destination` | Load and parse the MongoDB destination schema JSON |
| 3 | `discover_schemas` | LLM independently analyses each schema — field semantics, naming conventions, relationships, coded values |
| 4 | `ask_questions` | LLM generates 4-6 clarifying questions about ambiguities and edge cases |
| 5 | `user_input` | CLI terminal collects human responses |
| 6 | `map_tables` | LLM maps source tables → destination collections with confidence + reasoning |
| 7 | `map_fields` | For each table pair, LLM maps every field with confidence, type_transform, reasoning, and notes |
| 8 | `human_review` | Routes low-confidence mappings (<0.8) to user for confirmation/correction |
| 9 | `generate_output` | Assembles final JSON output matching the required schema |

---

## Project Structure

```text
DhanyaInterviewTask/
│
├── graph/                          Core pipeline (LangGraph + LangChain)
│   ├── __init__.py
│   ├── state.py                    — TypedDict state definition
│   ├── schema_tools.py             — LangChain @tool functions for LLM reasoning
│   ├── mapper_graph.py             — LangGraph StateGraph + nodes + routing + CLI
│   └── draw_graph.py               — Graph visualisation utility
│
├── schemas/                        Input schemas
│   ├── source_schema.json          — MySQL legacy_hrm (Dataset A)
│   └── destination_schema.json     — MongoDB people_platform (Dataset B)
│
├── tests/                          Unit tests
│   ├── __init__.py
│   └── test_schema_tools.py        — Tool output structure tests (mocked LLM)
│
├── output/                         Generated output (git-ignored)
│   └── mapping_output.json         — Final field mapping JSON
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env
```

### 3. Run the Pipeline

```bash
python3 -m graph.mapper_graph schemas/source_schema.json schemas/destination_schema.json
```

Optional: set a custom confidence threshold (default 0.8):
```bash
python3 -m graph.mapper_graph schemas/source_schema.json schemas/destination_schema.json 0.9
```

### 4. Run Tests

```bash
python3 -m pytest tests/ -v
```

### 5. Visualise the Graph

```bash
python3 -m graph.draw_graph
```

---

## Output Format

The pipeline produces a JSON file at `output/mapping_output.json`:

```json
{
  "mapping_version": "1.0",
  "source": "legacy_hrm (MySQL)",
  "destination": "people_platform (MongoDB)",
  "generated_at": "<ISO 8601 timestamp>",
  "tables": [
    {
      "source_table": "emp_master",
      "destination_collection": "employees",
      "confidence": 0.97,
      "reasoning": "Both represent the core employee entity ...",
      "field_mappings": [
        {
          "source_field": "f_name",
          "destination_field": "fullName.firstName",
          "type_transform": "VARCHAR -> String (nested path)",
          "confidence": 0.98,
          "reasoning": "Flat field promoted into the fullName sub-document.",
          "notes": null
        }
      ],
      "unmapped_source_fields": [],
      "unmapped_destination_fields": []
    }
  ]
}
```

---

## Design Decisions

### Critical Constraint Compliance
Both schemas are **never passed in a single LLM prompt**. The pipeline:
1. Analyses the source schema independently (`analyze_source_schema` tool)
2. Analyses the destination schema independently (`analyze_destination_schema` tool)
3. Uses the **analysed outputs** (not raw schemas) for table and field mapping

### Multi-Step Agentic Workflow
Instead of a single LLM call, the pipeline uses **9 orchestrated nodes** in a LangGraph StateGraph. This enables:
- **Confidence gating** — auto-approve high-confidence mappings, route low-confidence to humans
- **Iterative refinement** — retry with feedback on failed or ambiguous mappings
- **Human-in-the-loop** — pause for CLI input at discovery and review steps

### Field-Level Granularity
Each field mapping includes:
- **Dot notation** for nested MongoDB paths (e.g., `fullName.firstName`)
- **Type transformation** descriptions (e.g., `TINYINT(1) -> Boolean`)
- **Value transform notes** for coded fields (e.g., `A -> active, I -> inactive, T -> terminated`)
- **Confidence scores** calibrated per-field

### Prompt Engineering
Prompts are structured with:
- **System message**: Role definition, output schema, constraints
- **Human message**: Only the relevant data for that step
- **Temperature 0.0**: Deterministic outputs for reproducibility
- **JSON-only output**: Structured responses parsed with fence-aware parser
