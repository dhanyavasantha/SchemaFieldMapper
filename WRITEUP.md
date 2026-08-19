# Schema Field Mapper — Design & Prompt Engineering Write-up

## 1. Problem Overview & Critical Constraint
The goal is to autonomously map every field from a relational MySQL database (`legacy_hrm`) to a document-oriented MongoDB schema (`people_platform`). 

### Critical Constraint
> **You cannot pass both schemas to an LLM in a single prompt and receive a finished mapping.**

Passing entire source and destination schemas together creates attention dispersion, hallucinations in nested structures, and violates the single-prompt constraint. To solve this, we architected a cyclic state machine using **LangGraph** with granular **LangChain** `@tool` reasoning steps.

---

## 2. Architecture & Pipeline Breakdown

```
[Ingest Source] ──► [Ingest Destination] ──► [Discover Schemas (Independent)]
                                                       │
[Map Fields (Table-by-Table)] ◄── [Map Tables] ◄── [User Input] ◄── [Ask Questions]
           │
     (Confidence Gate)
      ├── ≥ 80% ──► Next Table / Generate Output
      └── < 80% ──► [Human Review / Refine] ──► Next Table
                                                       │
                                            [Generate Output JSON]
```

### The 7 Stages:
1. **Schema Ingestion (`ingest_source`, `ingest_destination`)**: Parses raw JSON schemas, registers table and collection names into state.
2. **Independent Discovery (`discover_schemas`)**: Two separate LLM calls analyze Dataset A and Dataset B in isolation:
   - Source analyzer extracts constraints, PK/FK relationships, abbreviation patterns (`f_name`, `emp_cd`), and coded enums (`rec_stat`).
   - Destination analyzer extracts nested sub-documents (`fullName`, `employment`, `compensation`), dot-notation paths, and reference ObjectIds.
3. **Clarifying Questions (`ask_questions` & `user_input`)**: Synthesizes the two analyses to formulate targeted questions about ambiguities (e.g., how to handle `rec_stat` single-char codes or unmapped fields like `dob`).
4. **Table-Level Mapping (`map_tables`)**: Maps source tables to destination collections (e.g., `emp_master` → `employees`, `dept_info` → `departments`, `locations` → `locations`) with confidence scoring.
5. **Field-by-Field Mapping (`map_fields`)**: Iterates through each table pair one by one. The LLM only receives the focused analysis of that specific table pair, generating:
   - `destination_field` in dot notation
   - `type_transform`
   - `confidence` (0.0 to 1.0)
   - `reasoning`
   - `notes` (e.g., transformation lookups)
6. **Confidence Gating & Human Review (`human_review`)**: Mappings with confidence below the threshold (default `0.80`) are routed to human review. The user can provide corrections, which triggers a focused re-evaluation tool (`review_low_confidence`).
7. **Final JSON Generation (`generate_output`)**: Compiles the complete mapping into the exact required format and saves it to `output/mapping_output.json`.

---

## 3. Prompt Engineering Strategy

All prompts use strict separation of concerns, JSON output enforcement, and temperature `0.0` for deterministic execution:

### A. Independent Schema Analysis Prompts
- **Source Prompt (`analyze_source_schema`)**: Instructs the model to act as a relational database analyst. Focuses on PKs, FKs, column naming patterns, and enum codes without any knowledge of the target system.
- **Destination Prompt (`analyze_destination_schema`)**: Instructs the model to act as a document database specialist. Forces extraction of complete dot-notation paths (`fullName.firstName`, `employment.status`) and sub-document hierarchies.

### B. Discovery Questions Prompt (`generate_discovery_questions`)
- Takes the summarized structural findings and prompts the LLM to identify high-risk migration edge cases:
  1. Coded value lookups (`rec_stat` → `A/I/T`)
  2. Data type impedance mismatches (`INT` PKs → `ObjectId`s)
  3. Denormalization / nested document structures
  4. Missing or unmapped attributes (`dob`, `job_lvl_cd`)

### C. Isolated Field Mapping Prompt (`map_fields_for_table`)
- Constrained strictly to one table pair (`source_table_name` and `destination_collection_name`).
- Enforces the schema rule: every single source field must be evaluated with explicit type transformation and reasoning.

---

## 4. Key Design Decisions

1. **Zero Governance / Zero Web Overhead**: Stripped of FastAPI, RMF governance layers, and grouped-layout normalizers to keep the execution lightweight, fast, and maintainable.
2. **Dot-Notation Path Standard**: Standardized on dot-notation (e.g. `fullName.firstName`, `employment.jobLevel`) to naturally map flat relational columns into MongoDB nested sub-documents.
3. **Resilient Execution & CLI Portability**: Embedded automatic `sys.path` resolution so scripts can be invoked directly from the project root or within the `graph/` subdirectory.
4. **Mocked Unit Test Suite**: Comprehensive pytest suite (19 tests) testing tool outputs, JSON schema compliance, and confidence score bounds without incurring LLM token costs.
