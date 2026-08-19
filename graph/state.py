"""
state.py
--------
TypedDict state definition for the Schema Field Mapper LangGraph pipeline.
"""

from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SchemaMapperState(TypedDict):
    """Full state flowing through the LangGraph schema mapping pipeline."""

    # ── Inputs ────────────────────────────────────────────────────────────
    source_schema_path: str
    destination_schema_path: str

    # ── Raw schemas (loaded once) ─────────────────────────────────────────
    source_schema_raw: str                     # raw JSON string
    destination_schema_raw: str                # raw JSON string

    # ── LLM analyses (one per schema, never combined) ─────────────────────
    source_analysis: str                       # LLM structured analysis of source
    destination_analysis: str                  # LLM structured analysis of destination

    # ── User interaction ──────────────────────────────────────────────────
    user_hints: str                            # accumulated user hints
    pending_question: str                      # question awaiting user response
    user_response: str                         # latest user answer
    current_step: str                          # tracks which step we're on

    # ── Table-level mapping ───────────────────────────────────────────────
    table_mapping: list                        # [{source_table, dest_collection, confidence, reasoning}]

    # ── Field-level mapping (iterative, one table pair at a time) ─────────
    current_table_index: int                   # index into table_mapping being processed
    field_mappings: list                       # accumulated field mapping results per table pair
    low_confidence_fields: list                # fields below threshold for human review

    # ── Output ────────────────────────────────────────────────────────────
    output_json: Optional[dict]                # final JSON output matching required schema
    output_path: str                           # path to written output file

    # ── Control ───────────────────────────────────────────────────────────
    status: str                                # loading | discovering | asking | mapping_tables | mapping_fields | reviewing | done | failed
    retry_count: int
    feedback: str                              # retry feedback from user or prior step
    confidence_threshold: float                # auto-approve above this (default 0.8)

    # ── LangGraph messages ────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
