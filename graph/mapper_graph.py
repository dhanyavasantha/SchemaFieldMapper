"""
mapper_graph.py
---------------
LangGraph StateGraph orchestration for the Schema Field Mapper pipeline.

Workflow:
    ingest_source → ingest_destination → discover_schemas → ask_questions →
    user_input → map_tables → map_fields → (human_review loop) → generate_output

Uses:
    - LangGraph StateGraph with conditional edges
    - LangChain @tool calls for LLM reasoning
    - CLI human-in-the-loop via terminal input()
    - Confidence gating with configurable threshold
"""

import os
import sys
import json
from typing import Optional
from datetime import datetime, timezone

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage

from graph.state import SchemaMapperState
from graph.schema_tools import (
    analyze_source_schema,
    analyze_destination_schema,
    generate_discovery_questions,
    map_tables_to_collections,
    map_fields_for_table,
    review_low_confidence,
    parse_llm_json,
)

MAX_RETRIES = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.8


# ══════════════════════════════════════════════════════════════════════════════
# Node: Ingest Source Schema
# ══════════════════════════════════════════════════════════════════════════════
def ingest_source_node(state: SchemaMapperState) -> dict:
    """Load and store the raw source schema JSON."""
    print("\n" + "═" * 60)
    print("  STEP 1: INGEST SOURCE SCHEMA")
    print("═" * 60)

    path = state["source_schema_path"]
    with open(path, "r") as f:
        raw = f.read()

    schema = json.loads(raw)
    db_name = schema.get("database", "unknown")
    db_type = schema.get("db_type", schema.get("type", "unknown"))
    tables = list(schema.get("tables", schema.get("collections", {})).keys())

    print(f"  Database: {db_name} ({db_type})")
    print(f"  Tables: {', '.join(tables)}")

    return {
        "source_schema_raw": raw,
        "status": "ingesting_destination",
        "current_step": "ingest_destination",
        "messages": [AIMessage(content=f"Loaded source schema: {db_name} with tables {tables}")],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node: Ingest Destination Schema
# ══════════════════════════════════════════════════════════════════════════════
def ingest_destination_node(state: SchemaMapperState) -> dict:
    """Load and store the raw destination schema JSON."""
    print("\n" + "═" * 60)
    print("  STEP 2: INGEST DESTINATION SCHEMA")
    print("═" * 60)

    path = state["destination_schema_path"]
    with open(path, "r") as f:
        raw = f.read()

    schema = json.loads(raw)
    db_name = schema.get("database", "unknown")
    db_type = schema.get("db_type", schema.get("type", "unknown"))
    collections = list(schema.get("collections", schema.get("tables", {})).keys())

    print(f"  Database: {db_name} ({db_type})")
    print(f"  Collections: {', '.join(collections)}")

    return {
        "destination_schema_raw": raw,
        "status": "discovering",
        "current_step": "discover_schemas",
        "messages": [AIMessage(content=f"Loaded destination schema: {db_name} with collections {collections}")],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node: Discover Schemas (independent LLM analyses)
# ══════════════════════════════════════════════════════════════════════════════
def discover_schemas_node(state: SchemaMapperState) -> dict:
    """Run LLM analysis on source and destination schemas INDEPENDENTLY."""
    print("\n" + "═" * 60)
    print("  STEP 3: DISCOVER & ANALYSE SCHEMAS")
    print("  (Each schema analysed independently — critical constraint)")
    print("═" * 60)

    # ── Analyse source schema ─────────────────────────────────────────────
    print("\n  Analysing SOURCE schema...")
    source_analysis = analyze_source_schema.invoke({
        "schema_raw_text": state["source_schema_raw"],
    })
    source_parsed = parse_llm_json(source_analysis)
    source_tables = [t["table_name"] for t in source_parsed.get("tables", [])]
    naming = source_parsed.get("naming_conventions", {})
    print(f"    Tables identified: {source_tables}")
    if naming:
        print(f"    Naming conventions: {json.dumps(naming, indent=2)[:200]}")

    # ── Analyse destination schema ────────────────────────────────────────
    print("\n  Analysing DESTINATION schema...")
    destination_analysis = analyze_destination_schema.invoke({
        "schema_raw_text": state["destination_schema_raw"],
    })
    dest_parsed = parse_llm_json(destination_analysis)
    dest_collections = [c["collection_name"] for c in dest_parsed.get("collections", [])]
    print(f"    Collections identified: {dest_collections}")

    return {
        "source_analysis": source_analysis,
        "destination_analysis": destination_analysis,
        "status": "asking",
        "current_step": "ask_questions",
        "messages": [AIMessage(content=f"Schema discovery complete. Source: {source_tables}, Destination: {dest_collections}")],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node: Ask Initial Questions
# ══════════════════════════════════════════════════════════════════════════════
def ask_questions_node(state: SchemaMapperState) -> dict:
    """Generate clarifying questions based on schema analyses."""
    print("\n" + "═" * 60)
    print("  STEP 4: DISCOVERY QUESTIONS")
    print("═" * 60)

    source_schema = json.loads(state["source_schema_raw"])
    dest_schema = json.loads(state["destination_schema_raw"])

    questions = generate_discovery_questions.invoke({
        "source_analysis": state["source_analysis"],
        "destination_analysis": state["destination_analysis"],
        "source_db_name": source_schema.get("database", "source"),
        "destination_db_name": dest_schema.get("database", "destination"),
    })

    print(f"\n{questions}")

    return {
        "pending_question": questions,
        "current_step": "initial_questions",
        "status": "waiting_user",
        "messages": [AIMessage(content=questions)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node: User Input (CLI human-in-the-loop)
# ══════════════════════════════════════════════════════════════════════════════
def user_input_node(state: SchemaMapperState) -> dict:
    """Collect user input via terminal."""
    question = state.get("pending_question", "Any additional input?")

    print(f"\n{'─' * 40}")
    print(f"  ⏳ HUMAN INPUT REQUIRED")
    print(f"{'─' * 40}")
    print(f"\n{question}\n")

    user_response = input("Your response (or 'skip' to proceed): ").strip()

    if not user_response or user_response.lower() == "skip":
        user_response = "No additional input."

    # Accumulate hints
    existing_hints = state.get("user_hints", "")
    if user_response != "No additional input.":
        existing_hints += f"\n{user_response}" if existing_hints else user_response

    return {
        "user_response": user_response,
        "user_hints": existing_hints,
        "pending_question": "",
        "messages": [HumanMessage(content=user_response)],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node: Map Tables to Collections
# ══════════════════════════════════════════════════════════════════════════════
def map_tables_node(state: SchemaMapperState) -> dict:
    """Map source tables to destination collections."""
    print("\n" + "═" * 60)
    print("  STEP 5: MAP TABLES → COLLECTIONS")
    print("═" * 60)

    result_text = map_tables_to_collections.invoke({
        "source_analysis": state["source_analysis"],
        "destination_analysis": state["destination_analysis"],
        "user_hints": state.get("user_hints", ""),
    })

    result = parse_llm_json(result_text)
    table_mappings = result.get("table_mappings", [])

    for tm in table_mappings:
        conf = tm.get("confidence", 0)
        print(f"  {tm['source_table']} → {tm['destination_collection']}  (confidence: {conf:.0%})")
        print(f"    Reasoning: {tm.get('reasoning', 'N/A')}")

    unmapped_src = result.get("unmapped_source_tables", [])
    unmapped_dst = result.get("unmapped_destination_collections", [])
    if unmapped_src:
        print(f"\n  ⚠ Unmapped source tables: {unmapped_src}")
    if unmapped_dst:
        print(f"  ⚠ Unmapped destination collections: {unmapped_dst}")

    return {
        "table_mapping": table_mappings,
        "current_table_index": 0,
        "field_mappings": [],
        "status": "mapping_fields",
        "current_step": "map_fields",
        "messages": [AIMessage(content=f"Table mapping complete: {len(table_mappings)} pairs identified")],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node: Map Fields for Current Table Pair
# ══════════════════════════════════════════════════════════════════════════════
def map_fields_node(state: SchemaMapperState) -> dict:
    """Map every field in the current table pair."""
    idx = state.get("current_table_index", 0)
    table_mappings = state.get("table_mapping", [])
    accumulated = list(state.get("field_mappings", []))

    if idx >= len(table_mappings):
        # All tables processed
        return {
            "status": "generating",
            "current_step": "generate_output",
        }

    pair = table_mappings[idx]
    src_table = pair["source_table"]
    dst_collection = pair["destination_collection"]

    print(f"\n{'═' * 60}")
    print(f"  STEP 6.{idx + 1}: MAP FIELDS — {src_table} → {dst_collection}")
    print(f"{'═' * 60}")

    # ── Extract per-table analysis from full analyses ─────────────────────
    source_parsed = parse_llm_json(state["source_analysis"])
    dest_parsed = parse_llm_json(state["destination_analysis"])

    src_table_analysis = None
    for t in source_parsed.get("tables", []):
        if t.get("table_name", "").lower() == src_table.lower():
            src_table_analysis = json.dumps(t, indent=2)
            break
    if not src_table_analysis:
        src_table_analysis = state["source_analysis"]

    dst_collection_analysis = None
    for c in dest_parsed.get("collections", []):
        if c.get("collection_name", "").lower() == dst_collection.lower():
            dst_collection_analysis = json.dumps(c, indent=2)
            break
    if not dst_collection_analysis:
        dst_collection_analysis = state["destination_analysis"]

    # ── Call LLM for field mapping ────────────────────────────────────────
    result_text = map_fields_for_table.invoke({
        "source_table_name": src_table,
        "source_table_analysis": src_table_analysis,
        "destination_collection_name": dst_collection,
        "destination_collection_analysis": dst_collection_analysis,
        "user_hints": state.get("user_hints", ""),
        "feedback": state.get("feedback", ""),
    })

    result = parse_llm_json(result_text)
    field_maps = result.get("field_mappings", [])

    # ── Print results ─────────────────────────────────────────────────────
    threshold = state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    low_conf = []

    for fm in field_maps:
        conf = fm.get("confidence", 0)
        dest = fm.get("destination_field", "UNMAPPED")
        tag = "✓" if conf >= threshold else "⚠"
        print(f"  {tag} {fm['source_field']} → {dest}  (conf: {conf:.0%}, type: {fm.get('type_transform', 'N/A')})")
        if fm.get("notes"):
            print(f"       Notes: {fm['notes']}")
        if conf < threshold:
            low_conf.append(fm)

    unmapped_src = result.get("unmapped_source_fields", [])
    unmapped_dst = result.get("unmapped_destination_fields", [])
    if unmapped_src:
        print(f"\n  ⚠ Unmapped source fields: {unmapped_src}")
    if unmapped_dst:
        print(f"  ⚠ Unmapped destination fields: {unmapped_dst}")

    # ── Store results ─────────────────────────────────────────────────────
    table_result = {
        "source_table": src_table,
        "destination_collection": dst_collection,
        "confidence": pair.get("confidence", 0.9),
        "reasoning": pair.get("reasoning", ""),
        "field_mappings": field_maps,
        "unmapped_source_fields": unmapped_src,
        "unmapped_destination_fields": unmapped_dst,
    }
    accumulated.append(table_result)

    updates = {
        "field_mappings": accumulated,
        "low_confidence_fields": low_conf,
        "feedback": "",
        "messages": [AIMessage(content=f"Field mapping for {src_table} → {dst_collection}: {len(field_maps)} fields mapped, {len(low_conf)} low-confidence")],
    }

    if low_conf:
        # Route to human review
        low_conf_summary = "\n".join(
            f"  - {f['source_field']} → {f.get('destination_field', '?')} (conf: {f.get('confidence', 0):.0%})"
            for f in low_conf
        )
        updates["pending_question"] = (
            f"The following field mappings for {src_table} → {dst_collection} have LOW CONFIDENCE:\n"
            f"{low_conf_summary}\n\n"
            f"Please provide corrections or type 'approve' to accept as-is."
        )
        updates["status"] = "reviewing"
        updates["current_step"] = "human_review"
    else:
        # Move to next table
        updates["current_table_index"] = idx + 1
        updates["status"] = "mapping_fields"
        updates["current_step"] = "map_fields"

    return updates


# ══════════════════════════════════════════════════════════════════════════════
# Node: Human Review (for low-confidence mappings)
# ══════════════════════════════════════════════════════════════════════════════
def human_review_node(state: SchemaMapperState) -> dict:
    """Collect human feedback on low-confidence mappings and re-evaluate."""
    question = state.get("pending_question", "Review needed.")

    print(f"\n{'─' * 40}")
    print(f"  ⏳ HUMAN REVIEW REQUIRED")
    print(f"{'─' * 40}")
    print(f"\n{question}\n")

    user_feedback = input("Your feedback (or 'approve' to accept): ").strip()

    if not user_feedback or user_feedback.lower() == "approve":
        # Accept as-is, move to next table
        idx = state.get("current_table_index", 0)
        print("  ✓ Mappings approved.")
        return {
            "current_table_index": idx + 1,
            "low_confidence_fields": [],
            "pending_question": "",
            "status": "mapping_fields",
            "current_step": "map_fields",
            "messages": [HumanMessage(content="Approved low-confidence mappings")],
        }

    # ── Re-evaluate with LLM using user feedback ─────────────────────────
    idx = state.get("current_table_index", 0)
    table_mappings = state.get("table_mapping", [])
    pair = table_mappings[idx]

    low_conf = state.get("low_confidence_fields", [])
    low_conf_json = json.dumps(low_conf, indent=2)

    result_text = review_low_confidence.invoke({
        "low_confidence_mappings": low_conf_json,
        "user_feedback": user_feedback,
        "source_table_name": pair["source_table"],
        "destination_collection_name": pair["destination_collection"],
    })

    result = parse_llm_json(result_text)
    reviewed = result.get("reviewed_mappings", [])

    # ── Update the field mappings for this table ──────────────────────────
    accumulated = list(state.get("field_mappings", []))
    if accumulated:
        current_table_result = accumulated[-1]
        existing_mappings = current_table_result.get("field_mappings", [])

        # Replace low-confidence mappings with reviewed ones
        reviewed_sources = {r["source_field"] for r in reviewed}
        updated_mappings = [
            fm for fm in existing_mappings
            if fm["source_field"] not in reviewed_sources
        ]
        updated_mappings.extend(reviewed)
        current_table_result["field_mappings"] = updated_mappings
        accumulated[-1] = current_table_result

    for r in reviewed:
        action = r.get("review_action", "unknown")
        print(f"  {action.upper()}: {r['source_field']} → {r.get('destination_field', 'N/A')} (conf: {r.get('confidence', 0):.0%})")

    return {
        "field_mappings": accumulated,
        "current_table_index": idx + 1,
        "low_confidence_fields": [],
        "pending_question": "",
        "user_hints": state.get("user_hints", "") + f"\n{user_feedback}",
        "status": "mapping_fields",
        "current_step": "map_fields",
        "messages": [HumanMessage(content=f"Reviewed {len(reviewed)} low-confidence mappings")],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node: Generate Output JSON
# ══════════════════════════════════════════════════════════════════════════════
def generate_output_node(state: SchemaMapperState) -> dict:
    """Assemble final JSON output matching the required schema."""
    print("\n" + "═" * 60)
    print("  STEP 7: GENERATE OUTPUT")
    print("═" * 60)

    source_schema = json.loads(state["source_schema_raw"])
    dest_schema = json.loads(state["destination_schema_raw"])

    output = {
        "mapping_version": "1.0",
        "source": f"{source_schema['database']} ({source_schema.get('type', 'Unknown')})",
        "destination": f"{dest_schema['database']} ({dest_schema.get('type', 'Unknown')})",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": state.get("field_mappings", []),
    }

    # ── Write to file ─────────────────────────────────────────────────────
    output_dir = os.path.dirname(state["source_schema_path"])
    output_path = os.path.join(output_dir, "..", "output", "mapping_output.json")
    output_path = os.path.normpath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  ✓ Output written to: {output_path}")
    print(f"  Tables mapped: {len(output['tables'])}")
    total_fields = sum(len(t.get('field_mappings', [])) for t in output['tables'])
    print(f"  Total field mappings: {total_fields}")

    # ── Summary ───────────────────────────────────────────────────────────
    for table in output["tables"]:
        src = table.get("source_table", "?")
        dst = table.get("destination_collection", "?")
        n_fields = len(table.get("field_mappings", []))
        unmapped_src = len(table.get("unmapped_source_fields", []))
        unmapped_dst = len(table.get("unmapped_destination_fields", []))
        print(f"\n  {src} → {dst}: {n_fields} fields")
        if unmapped_src:
            print(f"    Unmapped source: {table['unmapped_source_fields']}")
        if unmapped_dst:
            print(f"    Unmapped destination: {table['unmapped_destination_fields']}")

    return {
        "output_json": output,
        "output_path": output_path,
        "status": "done",
        "current_step": "done",
        "messages": [AIMessage(content=f"Output generated: {total_fields} field mappings across {len(output['tables'])} tables → {output_path}")],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Routing Functions
# ══════════════════════════════════════════════════════════════════════════════
def route_after_questions(state: SchemaMapperState) -> str:
    """After asking questions, route to user input."""
    return "user_input"


def route_after_user_input(state: SchemaMapperState) -> str:
    """After user input, route to the appropriate next step."""
    step = state.get("current_step", "")
    if step == "initial_questions":
        return "map_tables"
    elif step == "human_review":
        return "human_review"
    return "map_tables"


def route_after_field_mapping(state: SchemaMapperState) -> str:
    """After field mapping, route based on confidence and remaining tables."""
    status = state.get("status", "")

    if status == "reviewing":
        return "human_review"
    elif status == "mapping_fields":
        # Check if all tables are processed
        idx = state.get("current_table_index", 0)
        table_mappings = state.get("table_mapping", [])
        if idx >= len(table_mappings):
            return "generate_output"
        return "map_fields"
    elif status == "generating":
        return "generate_output"

    return "generate_output"


def route_after_review(state: SchemaMapperState) -> str:
    """After human review, continue mapping or generate output."""
    idx = state.get("current_table_index", 0)
    table_mappings = state.get("table_mapping", [])
    if idx >= len(table_mappings):
        return "generate_output"
    return "map_fields"


# ══════════════════════════════════════════════════════════════════════════════
# Build Graph
# ══════════════════════════════════════════════════════════════════════════════
def build_schema_mapper_graph() -> StateGraph:
    """Build and compile the LangGraph schema mapper pipeline."""

    graph = StateGraph(SchemaMapperState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    graph.add_node("ingest_source", ingest_source_node)
    graph.add_node("ingest_destination", ingest_destination_node)
    graph.add_node("discover_schemas", discover_schemas_node)
    graph.add_node("ask_questions", ask_questions_node)
    graph.add_node("user_input", user_input_node)
    graph.add_node("map_tables", map_tables_node)
    graph.add_node("map_fields", map_fields_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("generate_output", generate_output_node)

    # ── Set entry point ───────────────────────────────────────────────────
    graph.set_entry_point("ingest_source")

    # ── Add edges ─────────────────────────────────────────────────────────
    # Linear flow: ingest_source → ingest_destination → discover → ask
    graph.add_edge("ingest_source", "ingest_destination")
    graph.add_edge("ingest_destination", "discover_schemas")
    graph.add_edge("discover_schemas", "ask_questions")

    # ask_questions → user_input (always)
    graph.add_conditional_edges("ask_questions", route_after_questions, {
        "user_input": "user_input",
    })

    # user_input → map_tables or human_review
    graph.add_conditional_edges("user_input", route_after_user_input, {
        "map_tables": "map_tables",
        "human_review": "human_review",
    })

    # map_tables → map_fields (always, first table pair)
    graph.add_edge("map_tables", "map_fields")

    # map_fields → human_review | map_fields (next table) | generate_output
    graph.add_conditional_edges("map_fields", route_after_field_mapping, {
        "human_review": "human_review",
        "map_fields": "map_fields",
        "generate_output": "generate_output",
    })

    # human_review → map_fields (next table) | generate_output
    graph.add_conditional_edges("human_review", route_after_review, {
        "map_fields": "map_fields",
        "generate_output": "generate_output",
    })

    # generate_output → END
    graph.add_edge("generate_output", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def run_schema_mapper(
    source_path: str,
    destination_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Run the full schema mapping pipeline.

    Args:
        source_path: Path to the source schema JSON file.
        destination_path: Path to the destination schema JSON file.
        confidence_threshold: Auto-approve threshold (default 0.8).

    Returns:
        dict with output_json, output_path, and status.
    """
    print("\n" + "═" * 60)
    print("  SCHEMA FIELD MAPPER — LangGraph Pipeline")
    print(f"  Source: {source_path}")
    print(f"  Destination: {destination_path}")
    print(f"  Confidence threshold: {confidence_threshold:.0%}")
    print("═" * 60)

    graph = build_schema_mapper_graph()

    initial_state: SchemaMapperState = {
        "source_schema_path": source_path,
        "destination_schema_path": destination_path,
        "source_schema_raw": "",
        "destination_schema_raw": "",
        "source_analysis": "",
        "destination_analysis": "",
        "user_hints": "",
        "pending_question": "",
        "user_response": "",
        "current_step": "ingest_source",
        "table_mapping": [],
        "current_table_index": 0,
        "field_mappings": [],
        "low_confidence_fields": [],
        "output_json": None,
        "output_path": "",
        "status": "loading",
        "retry_count": 0,
        "feedback": "",
        "confidence_threshold": confidence_threshold,
        "messages": [],
    }

    final_state = graph.invoke(initial_state)

    if final_state["status"] == "done":
        print(f"\n{'═' * 60}")
        print("  ✓ PIPELINE COMPLETE — Schema mapping generated successfully")
        print(f"    Output: {final_state.get('output_path', 'N/A')}")
        print(f"{'═' * 60}")
    else:
        print(f"\n{'═' * 60}")
        print(f"  ✗ PIPELINE ENDED — Status: {final_state['status']}")
        print(f"{'═' * 60}")

    return {
        "output_json": final_state.get("output_json"),
        "output_path": final_state.get("output_path", ""),
        "status": final_state.get("status", "failed"),
    }


# ── Terminal runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    default_src = os.path.join(PROJECT_ROOT, "schemas", "source_schema.json")
    default_dst = os.path.join(PROJECT_ROOT, "schemas", "destination_schema.json")

    if len(sys.argv) >= 3:
        src = sys.argv[1].strip(' "\'')
        dst = sys.argv[2].strip(' "\'')
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CONFIDENCE_THRESHOLD
    else:
        print(f"No schema paths provided. Using default schemas:\n  Source: {default_src}\n  Destination: {default_dst}\n")
        src = default_src
        dst = default_dst
        threshold = DEFAULT_CONFIDENCE_THRESHOLD

    # Resolve relative paths
    if not os.path.isabs(src) and not os.path.exists(src):
        src = os.path.join(PROJECT_ROOT, src)
    if not os.path.isabs(dst) and not os.path.exists(dst):
        dst = os.path.join(PROJECT_ROOT, dst)

    if not os.path.exists(src) or not os.path.exists(dst):
        print(f"Error: Could not locate schema files.\n  Source: {src}\n  Destination: {dst}")
        sys.exit(1)

    result = run_schema_mapper(src, dst, threshold)

    if result["status"] == "done":
        output = result["output_json"]
        total_fields = sum(len(t.get("field_mappings", [])) for t in output.get("tables", []))
        print(f"\n  Summary:")
        print(f"    Tables mapped: {len(output.get('tables', []))}")
        print(f"    Total field mappings: {total_fields}")
        print(f"    Output file: {result['output_path']}")
    else:
        print("\n  Pipeline failed.")
