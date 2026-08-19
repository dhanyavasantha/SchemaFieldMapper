"""
schema_tools.py
---------------
LangChain @tool-decorated functions for LLM-powered schema analysis and mapping.

Each tool wraps a ChatPromptTemplate | LLM chain and returns structured JSON.
Tools are designed to process schemas INDEPENDENTLY — the critical constraint
is that both schemas must never be passed in a single prompt.

Tools:
    analyze_source_schema      — Analyse source (MySQL) schema structure
    analyze_destination_schema — Analyse destination (MongoDB) schema structure
    generate_discovery_questions — Identify ambiguities and generate clarifying questions
    map_tables_to_collections  — Map source tables → destination collections
    map_fields_for_table       — Map every field in one table pair
    review_low_confidence      — Re-evaluate low-confidence mappings after human feedback
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()


# ── LLM factory ──────────────────────────────────────────────────────────────
def _get_llm():
    """Return a ChatOpenAI instance configured from environment."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
    )


def _invoke_chain(prompt: ChatPromptTemplate, variables: dict) -> str:
    """Invoke a prompt | LLM chain and return the content string.
    
    This is a thin wrapper that centralises the chain invocation,
    making it easy to mock in unit tests.
    """
    chain = prompt | _get_llm()
    response = chain.invoke(variables)
    return response.content


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Parse LLM JSON response safely
# ══════════════════════════════════════════════════════════════════════════════
def parse_llm_json(response_text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    raw = response_text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last fence lines
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        raw = "\n".join(lines[start:end])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        brace_start = raw.find("{")
        brace_end = raw.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(raw[brace_start:brace_end])
            except json.JSONDecodeError:
                return {}
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: Analyse Source Schema
# ══════════════════════════════════════════════════════════════════════════════
@tool
def analyze_source_schema(schema_raw_text: str) -> str:
    """Analyse a source database schema (MySQL/relational) to understand its
    structure, field semantics, relationships, data types, and constraints.
    Returns a structured JSON analysis."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database schema analyst. You are given a SOURCE database schema (MySQL / relational).

Your job is to produce a DETAILED structural analysis. For EACH table, analyse:
1. Table purpose — what entity does it represent?
2. Fields — for each field: semantic meaning, data type, constraints (PK, FK, UNIQUE, NOT NULL)
3. Relationships — foreign keys, self-references, join patterns
4. Data patterns — coded values (e.g. A=Active, I=Inactive), date formats, boolean representations
5. Naming conventions — abbreviations used (e.g. emp = employee, nm = name, dt = date, cd = code)

IMPORTANT: This is ONLY the source schema analysis. Do NOT reference any destination schema.

Respond ONLY with valid JSON:
{{
  "database": "<database name>",
  "db_type": "<database type>",
  "tables": [
    {{
      "table_name": "<name>",
      "purpose": "<what this table represents>",
      "row_count_hint": "<expected cardinality>",
      "fields": [
        {{
          "field_name": "<name>",
          "data_type": "<type>",
          "semantic_meaning": "<what this field represents in plain English>",
          "constraints": ["PK", "FK", "UNIQUE", "NOT NULL"],
          "fk_reference": "<table.field if FK, else null>",
          "coded_values": "<value mapping if coded, e.g. A=Active, else null>",
          "naming_pattern": "<abbreviation expansion, e.g. f_name = first name>"
        }}
      ],
      "relationships": ["<description of each relationship>"]
    }}
  ],
  "naming_conventions": {{
    "<abbreviation>": "<full meaning>"
  }}
}}"""),
        ("human", """Source schema:
{schema_raw_text}"""),
    ])

    return _invoke_chain(prompt, {"schema_raw_text": schema_raw_text})


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: Analyse Destination Schema
# ══════════════════════════════════════════════════════════════════════════════
@tool
def analyze_destination_schema(schema_raw_text: str) -> str:
    """Analyse a destination database schema (MongoDB/document) to understand its
    structure, nested documents, field semantics, references, and data types.
    Returns a structured JSON analysis."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a database schema analyst. You are given a DESTINATION database schema (MongoDB / document store).

Your job is to produce a DETAILED structural analysis. For EACH collection, analyse:
1. Collection purpose — what entity does it represent?
2. Fields — for each field (including nested): semantic meaning, data type, dot-notation path
3. Nested structures — identify sub-documents and their field groupings
4. References — ObjectId references to other collections
5. Data patterns — expected value formats (ISODate, Boolean, String enums)

IMPORTANT: List ALL fields using DOT NOTATION for nested paths (e.g. "fullName.firstName", "employment.status").

IMPORTANT: This is ONLY the destination schema analysis. Do NOT reference any source schema.

Respond ONLY with valid JSON:
{{
  "database": "<database name>",
  "db_type": "<database type>",
  "collections": [
    {{
      "collection_name": "<name>",
      "purpose": "<what this collection represents>",
      "fields": [
        {{
          "field_path": "<dot-notation path, e.g. fullName.firstName>",
          "data_type": "<type>",
          "semantic_meaning": "<what this field represents in plain English>",
          "is_nested": true/false,
          "parent_document": "<parent sub-document name or null>",
          "reference": "<collection._id if ref, else null>",
          "value_format": "<expected format details>"
        }}
      ],
      "sub_documents": ["<names of nested sub-document groupings>"],
      "references": ["<description of each reference>"]
    }}
  ]
}}"""),
        ("human", """Destination schema:
{schema_raw_text}"""),
    ])

    return _invoke_chain(prompt, {"schema_raw_text": schema_raw_text})


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: Generate Discovery Questions
# ══════════════════════════════════════════════════════════════════════════════
@tool
def generate_discovery_questions(
    source_analysis: str,
    destination_analysis: str,
    source_db_name: str,
    destination_db_name: str,
) -> str:
    """Based on independent analyses of source and destination schemas, generate
    smart clarifying questions about ambiguities, edge cases, and mapping decisions.
    Returns numbered questions as plain text."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data migration specialist preparing to map fields from a source database to a destination database.

You have received INDEPENDENT analyses of the source and destination schemas.
Your job is to identify ambiguities, edge cases, and key decisions that need human input.

Generate 4-6 SHORT, specific questions. Focus on:
1. Fields with CODED VALUES that need transformation (e.g. 'A' -> 'active')
2. Fields that exist in source but have NO OBVIOUS destination (or vice versa)
3. Type conversions that need clarification (e.g. INT primary key -> ObjectId)
4. Nested vs flat structure decisions
5. Fields where the semantic match is ambiguous (e.g. is 'dob' needed in destination?)
6. Reference/FK handling strategy

Start with a 1-line summary of the migration direction, then list numbered questions.
Keep each question to 1-2 sentences max."""),
        ("human", """Source database: {source_db_name}
Source analysis:
{source_analysis}

Destination database: {destination_db_name}
Destination analysis:
{destination_analysis}"""),
    ])

    return _invoke_chain(prompt, {
        "source_analysis": source_analysis,
        "destination_analysis": destination_analysis,
        "source_db_name": source_db_name,
        "destination_db_name": destination_db_name,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4: Map Tables to Collections
# ══════════════════════════════════════════════════════════════════════════════
@tool
def map_tables_to_collections(
    source_analysis: str,
    destination_analysis: str,
    user_hints: str = "",
) -> str:
    """Map source tables to destination collections based on semantic similarity.
    Returns JSON with table-level mappings, confidence scores, and reasoning."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are mapping source database TABLES to destination database COLLECTIONS.

Based on the independent analyses of source and destination schemas, determine which
source table maps to which destination collection.

Consider:
- Entity semantics (both represent employees, departments, locations, etc.)
- Field overlap and structural similarity
- Relationship patterns

{user_context}

Respond ONLY with valid JSON:
{{
  "table_mappings": [
    {{
      "source_table": "<table name>",
      "destination_collection": "<collection name>",
      "confidence": <0.0 to 1.0>,
      "reasoning": "<one sentence explaining the match>"
    }}
  ],
  "unmapped_source_tables": ["<tables with no destination match>"],
  "unmapped_destination_collections": ["<collections with no source match>"]
}}"""),
        ("human", """Source schema analysis:
{source_analysis}

Destination schema analysis:
{destination_analysis}"""),
    ])

    user_context = f"User hints: {user_hints}" if user_hints else ""

    return _invoke_chain(prompt, {
        "source_analysis": source_analysis,
        "destination_analysis": destination_analysis,
        "user_context": user_context,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Tool 5: Map Fields for a Single Table Pair
# ══════════════════════════════════════════════════════════════════════════════
@tool
def map_fields_for_table(
    source_table_name: str,
    source_table_analysis: str,
    destination_collection_name: str,
    destination_collection_analysis: str,
    user_hints: str = "",
    feedback: str = "",
) -> str:
    """Map EVERY field in a single source table to the corresponding destination
    collection field. Uses dot notation for nested paths.
    Returns JSON with field mappings, confidence, type_transform, reasoning, and notes."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are mapping fields from a SOURCE TABLE to a DESTINATION COLLECTION.

Source table: {source_table_name}
Destination collection: {destination_collection_name}

For EVERY source field, determine:
1. The destination field it maps to (use DOT NOTATION for nested paths, e.g. "fullName.firstName")
2. The type transformation required (e.g. "INT -> ObjectId", "TINYINT(1) -> Boolean", "CHAR(1) code -> String enum")
3. Your confidence in the mapping (0.0 to 1.0)
4. A brief reasoning sentence
5. Any value-transform notes (e.g. "Transform: A -> active, I -> inactive, T -> terminated"), or null

RULES:
- Map ALL source fields — do not skip any
- Use dot notation for nested destination paths
- If a source field has no destination equivalent, still include it with destination_field: null
- If a destination field has no source equivalent, list it in unmapped_destination_fields
- Be precise about type transformations — specify the exact conversion needed
- For coded values (e.g. CHAR(1) status codes), note the transform mapping in "notes"

{user_context}
{feedback_context}

Respond ONLY with valid JSON:
{{
  "source_table": "{source_table_name}",
  "destination_collection": "{destination_collection_name}",
  "field_mappings": [
    {{
      "source_field": "<field name>",
      "destination_field": "<dot-notation path or null>",
      "type_transform": "<source type -> destination type>",
      "confidence": <0.0 to 1.0>,
      "reasoning": "<one sentence>",
      "notes": "<transform logic or null>"
    }}
  ],
  "unmapped_source_fields": ["<source fields with no destination>"],
  "unmapped_destination_fields": ["<destination fields with no source>"]
}}"""),
        ("human", """Source table ({source_table_name}) analysis:
{source_table_analysis}

Destination collection ({destination_collection_name}) analysis:
{destination_collection_analysis}"""),
    ])

    user_context = f"User hints: {user_hints}" if user_hints else ""
    feedback_context = f"PREVIOUS ATTEMPT FEEDBACK: {feedback}" if feedback else ""

    return _invoke_chain(prompt, {
        "source_table_name": source_table_name,
        "destination_collection_name": destination_collection_name,
        "source_table_analysis": source_table_analysis,
        "destination_collection_analysis": destination_collection_analysis,
        "user_context": user_context,
        "feedback_context": feedback_context,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Tool 6: Review Low-Confidence Mappings
# ══════════════════════════════════════════════════════════════════════════════
@tool
def review_low_confidence(
    low_confidence_mappings: str,
    user_feedback: str,
    source_table_name: str,
    destination_collection_name: str,
) -> str:
    """Re-evaluate low-confidence field mappings after receiving human feedback.
    Returns updated mappings with revised confidence scores."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are reviewing field mappings that had LOW CONFIDENCE scores.

The user has provided feedback to help resolve these ambiguous mappings.
Re-evaluate each mapping considering the user's input.

For each mapping, either:
- CONFIRM the existing mapping with updated confidence
- CHANGE the destination field based on user feedback
- Mark as UNMAPPED if the user confirms no destination exists

Respond ONLY with valid JSON:
{{
  "reviewed_mappings": [
    {{
      "source_field": "<field name>",
      "destination_field": "<updated dot-notation path or null>",
      "type_transform": "<updated transform>",
      "confidence": <updated 0.0 to 1.0>,
      "reasoning": "<updated reasoning incorporating user feedback>",
      "notes": "<updated notes or null>",
      "review_action": "confirmed | changed | unmapped"
    }}
  ]
}}"""),
        ("human", """Source table: {source_table_name}
Destination collection: {destination_collection_name}

Low-confidence mappings to review:
{low_confidence_mappings}

User feedback:
{user_feedback}"""),
    ])

    return _invoke_chain(prompt, {
        "low_confidence_mappings": low_confidence_mappings,
        "user_feedback": user_feedback,
        "source_table_name": source_table_name,
        "destination_collection_name": destination_collection_name,
    })


# ══════════════════════════════════════════════════════════════════════════════
# All tools list
# ══════════════════════════════════════════════════════════════════════════════
SCHEMA_MAPPER_TOOLS = [
    analyze_source_schema,
    analyze_destination_schema,
    generate_discovery_questions,
    map_tables_to_collections,
    map_fields_for_table,
    review_low_confidence,
]
