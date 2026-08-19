"""
test_schema_tools.py
--------------------
Unit tests for the schema analysis and mapping tools.

Tests validate JSON output structure, confidence ranges, and field coverage.
Mocks _invoke_chain to avoid real LLM calls.
"""

import json
import pytest
from unittest.mock import patch
from graph.schema_tools import (
    parse_llm_json,
    analyze_source_schema,
    analyze_destination_schema,
    generate_discovery_questions,
    map_tables_to_collections,
    map_fields_for_table,
    review_low_confidence,
)

# Shorthand for the mock target
INVOKE_CHAIN = "graph.schema_tools._invoke_chain"


# ══════════════════════════════════════════════════════════════════════════════
# Test: parse_llm_json
# ══════════════════════════════════════════════════════════════════════════════
class TestParseLLMJson:
    """Test the JSON parser that handles markdown code fences."""

    def test_plain_json(self):
        raw = '{"key": "value", "num": 42}'
        result = parse_llm_json(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_with_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_json_with_plain_code_fence(self):
        raw = '```\n{"key": "value"}\n```'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"key": "value"}\nDone.'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_invalid_json_returns_empty(self):
        raw = "not json at all"
        result = parse_llm_json(raw)
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = parse_llm_json("")
        assert result == {}

    def test_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = parse_llm_json(raw)
        assert result["outer"]["inner"] == [1, 2, 3]
        assert result["flag"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Test: Source schema analysis output structure
# ══════════════════════════════════════════════════════════════════════════════
MOCK_SOURCE_ANALYSIS = json.dumps({
    "database": "legacy_hrm",
    "db_type": "MySQL (Relational)",
    "tables": [
        {
            "table_name": "emp_master",
            "purpose": "Core employee records",
            "row_count_hint": "thousands",
            "fields": [
                {
                    "field_name": "emp_id",
                    "data_type": "INT",
                    "semantic_meaning": "Unique employee identifier",
                    "constraints": ["PK"],
                    "fk_reference": None,
                    "coded_values": None,
                    "naming_pattern": "emp = employee, id = identifier",
                }
            ],
            "relationships": ["emp_master.dept_id -> dept_info.dept_id"],
        }
    ],
    "naming_conventions": {"emp": "employee", "nm": "name", "dt": "date"},
})


class TestAnalyzeSourceSchema:
    """Test that analyze_source_schema returns properly structured output."""

    @patch(INVOKE_CHAIN, return_value=MOCK_SOURCE_ANALYSIS)
    def test_output_structure(self, mock_invoke):
        result = analyze_source_schema.invoke({"schema_raw_text": '{"database": "test"}'})
        parsed = parse_llm_json(result)

        assert "database" in parsed
        assert "tables" in parsed
        assert len(parsed["tables"]) > 0
        assert "table_name" in parsed["tables"][0]
        assert "fields" in parsed["tables"][0]
        assert "naming_conventions" in parsed
        mock_invoke.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Test: Destination schema analysis output structure
# ══════════════════════════════════════════════════════════════════════════════
MOCK_DEST_ANALYSIS = json.dumps({
    "database": "people_platform",
    "db_type": "MongoDB (Document)",
    "collections": [
        {
            "collection_name": "employees",
            "purpose": "Employee profiles",
            "fields": [
                {
                    "field_path": "fullName.firstName",
                    "data_type": "String",
                    "semantic_meaning": "Employee first name",
                    "is_nested": True,
                    "parent_document": "fullName",
                    "reference": None,
                    "value_format": "Plain text string",
                }
            ],
            "sub_documents": ["fullName", "employment", "compensation"],
            "references": ["department.departmentId -> departments._id"],
        }
    ],
})


class TestAnalyzeDestinationSchema:
    """Test that analyze_destination_schema returns properly structured output."""

    @patch(INVOKE_CHAIN, return_value=MOCK_DEST_ANALYSIS)
    def test_output_structure(self, mock_invoke):
        result = analyze_destination_schema.invoke({"schema_raw_text": '{"database": "test"}'})
        parsed = parse_llm_json(result)

        assert "database" in parsed
        assert "collections" in parsed
        assert len(parsed["collections"]) > 0
        assert "collection_name" in parsed["collections"][0]
        assert "fields" in parsed["collections"][0]
        mock_invoke.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Test: Discovery questions
# ══════════════════════════════════════════════════════════════════════════════
MOCK_QUESTIONS = (
    "Migrating legacy_hrm (MySQL) → people_platform (MongoDB).\n\n"
    "1. How should rec_stat codes (A/I/T) be mapped to status strings?\n"
    "2. Should dob be preserved even though the destination has no date of birth field?\n"
    "3. How should MySQL INT primary keys be converted to MongoDB ObjectIds?\n"
    "4. Should department and location data be denormalized into the employees collection?\n"
)


class TestGenerateDiscoveryQuestions:
    """Test that generate_discovery_questions returns text with numbered questions."""

    @patch(INVOKE_CHAIN, return_value=MOCK_QUESTIONS)
    def test_returns_questions(self, mock_invoke):
        result = generate_discovery_questions.invoke({
            "source_analysis": "{}",
            "destination_analysis": "{}",
            "source_db_name": "legacy_hrm",
            "destination_db_name": "people_platform",
        })

        assert "1." in result
        assert "rec_stat" in result or "dob" in result or "ObjectId" in result
        mock_invoke.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Test: Table mapping output structure
# ══════════════════════════════════════════════════════════════════════════════
MOCK_TABLE_MAPPING = json.dumps({
    "table_mappings": [
        {
            "source_table": "emp_master",
            "destination_collection": "employees",
            "confidence": 0.97,
            "reasoning": "Both represent the core employee entity",
        },
        {
            "source_table": "dept_info",
            "destination_collection": "departments",
            "confidence": 0.95,
            "reasoning": "Both represent department/org units",
        },
        {
            "source_table": "locations",
            "destination_collection": "locations",
            "confidence": 0.99,
            "reasoning": "Direct name match, both represent physical locations",
        },
    ],
    "unmapped_source_tables": [],
    "unmapped_destination_collections": [],
})


class TestMapTablesToCollections:
    """Test that map_tables_to_collections returns properly structured output."""

    @patch(INVOKE_CHAIN, return_value=MOCK_TABLE_MAPPING)
    def test_output_structure(self, mock_invoke):
        result = map_tables_to_collections.invoke({
            "source_analysis": "{}",
            "destination_analysis": "{}",
            "user_hints": "",
        })
        parsed = parse_llm_json(result)

        assert "table_mappings" in parsed
        assert len(parsed["table_mappings"]) == 3

        for mapping in parsed["table_mappings"]:
            assert "source_table" in mapping
            assert "destination_collection" in mapping
            assert "confidence" in mapping
            assert 0.0 <= mapping["confidence"] <= 1.0
            assert "reasoning" in mapping

    @patch(INVOKE_CHAIN, return_value=MOCK_TABLE_MAPPING)
    def test_confidence_range(self, mock_invoke):
        result = map_tables_to_collections.invoke({
            "source_analysis": "{}",
            "destination_analysis": "{}",
        })
        parsed = parse_llm_json(result)

        for mapping in parsed["table_mappings"]:
            assert 0.0 <= mapping["confidence"] <= 1.0

    @patch(INVOKE_CHAIN, return_value=MOCK_TABLE_MAPPING)
    def test_all_three_tables_mapped(self, mock_invoke):
        result = map_tables_to_collections.invoke({
            "source_analysis": "{}",
            "destination_analysis": "{}",
        })
        parsed = parse_llm_json(result)

        source_tables = {m["source_table"] for m in parsed["table_mappings"]}
        assert source_tables == {"emp_master", "dept_info", "locations"}


# ══════════════════════════════════════════════════════════════════════════════
# Test: Field mapping output structure
# ══════════════════════════════════════════════════════════════════════════════
MOCK_FIELD_MAPPING = json.dumps({
    "source_table": "emp_master",
    "destination_collection": "employees",
    "field_mappings": [
        {
            "source_field": "emp_id",
            "destination_field": "_id",
            "type_transform": "INT -> ObjectId",
            "confidence": 0.91,
            "reasoning": "Primary key maps to MongoDB _id",
            "notes": "Store original emp_id as legacy field for traceability",
        },
        {
            "source_field": "f_name",
            "destination_field": "fullName.firstName",
            "type_transform": "VARCHAR -> String (nested path)",
            "confidence": 0.98,
            "reasoning": "Flat field promoted into fullName sub-document",
            "notes": None,
        },
        {
            "source_field": "rec_stat",
            "destination_field": "employment.status",
            "type_transform": "CHAR(1) code -> String enum",
            "confidence": 0.95,
            "reasoning": "Single-char codes require lookup transform",
            "notes": "Transform: A -> active, I -> inactive, T -> terminated",
        },
        {
            "source_field": "is_remote",
            "destination_field": "employment.isRemote",
            "type_transform": "TINYINT(1) -> Boolean",
            "confidence": 0.99,
            "reasoning": "MySQL boolean integer pattern maps to Boolean",
            "notes": None,
        },
    ],
    "unmapped_source_fields": [],
    "unmapped_destination_fields": [],
})


class TestMapFieldsForTable:
    """Test that map_fields_for_table returns properly structured output."""

    @patch(INVOKE_CHAIN, return_value=MOCK_FIELD_MAPPING)
    def test_output_structure(self, mock_invoke):
        result = map_fields_for_table.invoke({
            "source_table_name": "emp_master",
            "source_table_analysis": "{}",
            "destination_collection_name": "employees",
            "destination_collection_analysis": "{}",
        })
        parsed = parse_llm_json(result)

        assert "source_table" in parsed
        assert "destination_collection" in parsed
        assert "field_mappings" in parsed

        for fm in parsed["field_mappings"]:
            assert "source_field" in fm
            assert "destination_field" in fm
            assert "type_transform" in fm
            assert "confidence" in fm
            assert "reasoning" in fm
            assert "notes" in fm

    @patch(INVOKE_CHAIN, return_value=MOCK_FIELD_MAPPING)
    def test_dot_notation_in_destination(self, mock_invoke):
        result = map_fields_for_table.invoke({
            "source_table_name": "emp_master",
            "source_table_analysis": "{}",
            "destination_collection_name": "employees",
            "destination_collection_analysis": "{}",
        })
        parsed = parse_llm_json(result)

        nested_fields = [
            fm for fm in parsed["field_mappings"]
            if fm["destination_field"] and "." in fm["destination_field"]
        ]
        assert len(nested_fields) > 0, "Should have at least one nested dot-notation field"

    @patch(INVOKE_CHAIN, return_value=MOCK_FIELD_MAPPING)
    def test_required_output_fields(self, mock_invoke):
        """Every field mapping must have all 6 required fields."""
        result = map_fields_for_table.invoke({
            "source_table_name": "emp_master",
            "source_table_analysis": "{}",
            "destination_collection_name": "employees",
            "destination_collection_analysis": "{}",
        })
        parsed = parse_llm_json(result)

        required_keys = {"source_field", "destination_field", "type_transform",
                         "confidence", "reasoning", "notes"}
        for fm in parsed["field_mappings"]:
            assert required_keys.issubset(fm.keys()), f"Missing keys in {fm}"

    @patch(INVOKE_CHAIN, return_value=MOCK_FIELD_MAPPING)
    def test_type_transforms_present(self, mock_invoke):
        """All field mappings should have non-empty type transforms."""
        result = map_fields_for_table.invoke({
            "source_table_name": "emp_master",
            "source_table_analysis": "{}",
            "destination_collection_name": "employees",
            "destination_collection_analysis": "{}",
        })
        parsed = parse_llm_json(result)

        for fm in parsed["field_mappings"]:
            assert fm["type_transform"], f"Missing type_transform for {fm['source_field']}"


# ══════════════════════════════════════════════════════════════════════════════
# Test: Review low confidence output structure
# ══════════════════════════════════════════════════════════════════════════════
MOCK_REVIEW = json.dumps({
    "reviewed_mappings": [
        {
            "source_field": "dob",
            "destination_field": None,
            "type_transform": "N/A",
            "confidence": 0.9,
            "reasoning": "User confirmed dob has no destination equivalent",
            "notes": None,
            "review_action": "unmapped",
        }
    ],
})


class TestReviewLowConfidence:
    """Test that review_low_confidence returns properly structured output."""

    @patch(INVOKE_CHAIN, return_value=MOCK_REVIEW)
    def test_output_structure(self, mock_invoke):
        result = review_low_confidence.invoke({
            "low_confidence_mappings": "[]",
            "user_feedback": "dob is not needed",
            "source_table_name": "emp_master",
            "destination_collection_name": "employees",
        })
        parsed = parse_llm_json(result)

        assert "reviewed_mappings" in parsed
        for rm in parsed["reviewed_mappings"]:
            assert "review_action" in rm
            assert rm["review_action"] in ("confirmed", "changed", "unmapped")

    @patch(INVOKE_CHAIN, return_value=MOCK_REVIEW)
    def test_review_actions_valid(self, mock_invoke):
        result = review_low_confidence.invoke({
            "low_confidence_mappings": "[]",
            "user_feedback": "dob is not needed",
            "source_table_name": "emp_master",
            "destination_collection_name": "employees",
        })
        parsed = parse_llm_json(result)

        valid_actions = {"confirmed", "changed", "unmapped"}
        for rm in parsed["reviewed_mappings"]:
            assert rm["review_action"] in valid_actions
