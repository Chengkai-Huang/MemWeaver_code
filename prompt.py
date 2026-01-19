from __future__ import annotations
from typing import Any, Dict, Optional, List

def extract_time_prompt(dialogue_text: str, relation_desc: str):
    prompt = f"""
You are a time expression extractor.

Your Task:
Extract an ABSOLUTE time expression indicating when the event described by the given relation occurred or is scheduled to occur;
The absolute time may be inferred from relative time expressions in the dialogue.

Guidelines:
- Identify the time MOST relevant to the given relation/event.
- The dialogue may contain:
  - Absolute time expressions (e.g., "20 May 2022", "May 2022", "2022")
  - Relative time expressions (e.g., "yesterday", "last week", "this weekend")
- If a relative time expression is present, you MAY use it as a clue and resolve it into an ABSOLUTE time.
- The FINAL output MUST be an ABSOLUTE, HUMAN-READABLE time expression.

What counts as HUMAN-READABLE ABSOLUTE time:
- A specific calendar date (e.g., "20 May, 2022")
- A specific month and year (e.g., "May, 2022")
- A specific year (e.g., "2022")

DO NOT OUTPUT:
- Relative time expressions (e.g., "yesterday", "last week", "tomorrow")
- Vague time references without a calendar anchor (e.g., "recently", "soon", "later")

If there is NO clear usable time information for the target relation, return an empty string "".

Output format MUST be one of:
- Day-level:   "20 May, 2022"
- Month-level: "May, 2022"
- Year-level:  "2022"

Given:
- Dialogue: {dialogue_text}
- Relation: {relation_desc}

Output STRICT JSON:
{{
  "absolute_time": ""
}}""".strip()




    response_format= {
        "type": "json_schema",
        "json_schema": {
            "name": "relation_time_extraction",
            "schema": {
                "type": "object",
                "properties": {
                    "absolute_time": {
                        "type": "string",
                    }
                },
                "required": ["absolute_time"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    return prompt, response_format


def extract_ent_prompt(dialogue_text: str) -> tuple[str, dict]:
    prompt = f"""
You are an entity extraction assistant.

Your task:
Extract entities from a dialogue snippet.

Guidelines:
- Extract concise entity names that appear in the text (people,  locations, organizations, events, objects, etc.).
- Do not invent entities.
- Prefer a canonical form of the entity name.
- if an entity is expressed as a combined phrase, keep the full phrase intact and do not split it into smaller parts
- Return a de-duplicated list.

DO NOT EXTRACT:
1) Unclear entities such as "he", "that", "there".
2) Time/Date expressions (relative or absolute) such as "yesterday", "last week", "next month".

Dialogue:
{dialogue_text}

Output STRICT JSON:
{{
  "entities": ["entity1", "entity2", ...]
}}
""".strip()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "kg_entity_extraction_simple",
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["entities"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    return prompt, response_format


def extract_rel_prompt(dialogue_text: str, entity_list_text: str) -> tuple[str, dict]:
    prompt = f"""
You are a relation extraction assistant.

Your task:
Extract meaningful relations between entities as triples from a dialogue snippet.

Guidelines:
- Extract relations ONLY between entities in the provided list.
- Each relation must be directly supported by the text.
- Relation_type should be a short predicate phrase (lowercase preferred).
- If the relation happens under certain conditions, you can optionally include a "condition" field to describe it briefly.

DO NOT EXTRACT:
- time/location
- unmeaningful or vague relations for example "is related to", "has something to do with", "is associated with", etc.

Dialogue:
{dialogue_text}

Detected entities:
{entity_list_text}

Output STRICT JSON:
{{
"relations": [
    {{
    "source": "entity_name1",
    "target": "entity_name2",
    "relation_type": "short relation phrase",
    "condition": "if mentioned",
    }}
]
}}
""".strip()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "kg_relation_extraction_simple",
            "schema": {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "relation_type": {"type": "string"},
                                "condition": {"type": "string"},
                            },
                            "required": [
                                "source",
                                "target",
                                "relation_type",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["relations"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    return prompt, response_format


def review_kg_prompt(full_dialogue_text: str, entities_text: str, relations_text: str,dialogue_timestamp: str) -> tuple[str, dict]:
    prompt = f"""
You are a knowledge graph review assistant.

Your tasks:
Review a knowledge graph extracted from a dialogue session.

Options:
1) ADD:Find important relations that are clearly expressed in the dialogue but are MISSING from the current relation list.
2) UPDATE:For some EXISTING relations, refine relation_type, time/condition metadata.
3) DENY:If a relation in the current list is clearly NOT supported or contradicted by the dialogue, mark it as denied (to be removed).

You are given:
The full dialogue text for this session:

The dialogue hapeed at: {dialogue_timestamp}
{full_dialogue_text}
----------------

Existing entities in the KG for this session:
{entities_text}
----------------

Existing relations (triples) in the KG for this session:
{relations_text}
----------------

Output STRICT JSON:
{{
  "add": [
    {{"source": "A", "relation_type": "predicate", "target": "B", "time": "if mentioned", "condition": "if mentioned"}}
  ],
  "update": [
    {{"relation_id": "rid", "relation_type": "new predicate", "time": "if mentioned", "condition": "if mentioned"}}
  ],
  "deny": [
    {{"relation_id": "rid"}}
  ]
}}
""".strip()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "kg_session_review",
            "schema": {
                "type": "object",
                "properties": {
                    "add": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "relation_type": {"type": "string"},
                                "time": {"type": "string"},
                                "condition": {"type": "string"},
                            },
                            "required": ["source", "target", "relation_type"],
                            "additionalProperties": False,
                        },
                    },
                    "update": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation_id": {"type": "string"},
                                "relation_type": {"type": "string"},
                                "time": {"type": "string"},
                                "condition": {"type": "string"},
                            },
                            "required": ["relation_id"],
                            "additionalProperties": False,
                        },
                    },
                    "deny": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "relation_id": {"type": "string"},
                            },
                            "required": ["relation_id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["add", "update", "deny"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    return prompt, response_format




def _schema_to_text(schema: Dict[str, Any], indent: int = 0) -> str:
    """
    把 JSON Schema(子树) 递归转成可读的约束说明（不依赖 response_format 功能）。
    """
    pad = "  " * indent
    t = schema.get("type", "object")

    # 处理 anyOf / oneOf / allOf（简单版：列出分支）
    for comb in ("anyOf", "oneOf", "allOf"):
        if comb in schema and isinstance(schema[comb], list):
            lines = [f"{pad}- {comb}:"]
            for i, sub in enumerate(schema[comb], 1):
                lines.append(f"{pad}  - option {i}:")
                lines.append(_schema_to_text(sub, indent + 2))
            return "\n".join(lines)

    if t == "object":
        props: Dict[str, Any] = schema.get("properties", {}) or {}
        required: List[str] = schema.get("required", []) or []
        addl = schema.get("additionalProperties", None)

        lines = [f"{pad}- type: object"]
        if required:
            lines.append(f"{pad}- required keys: {required}")
        if addl is False:
            lines.append(f"{pad}- additionalProperties: false (禁止输出未声明字段)")
        elif isinstance(addl, dict):
            lines.append(f"{pad}- additionalProperties schema:")
            lines.append(_schema_to_text(addl, indent + 1))

        if props:
            lines.append(f"{pad}- properties:")
            for k, v in props.items():
                desc = v.get("description")
                lines.append(f"{pad}  - {k}:")
                lines.append(_schema_to_text(v, indent + 2))
                if desc:
                    lines.append(f"{pad}    - description: {desc}")
        return "\n".join(lines)

    if t == "array":
        items = schema.get("items", {})
        lines = [f"{pad}- type: array"]
        if "minItems" in schema:
            lines.append(f"{pad}- minItems: {schema['minItems']}")
        if "maxItems" in schema:
            lines.append(f"{pad}- maxItems: {schema['maxItems']}")
        lines.append(f"{pad}- items:")
        lines.append(_schema_to_text(items, indent + 1))
        return "\n".join(lines)

    # string/number/integer/boolean/null
    lines = [f"{pad}- type: {t}"]
    if "enum" in schema:
        lines.append(f"{pad}- enum: {schema['enum']}")
    if "pattern" in schema:
        lines.append(f"{pad}- pattern: {schema['pattern']}")
    if "minLength" in schema:
        lines.append(f"{pad}- minLength: {schema['minLength']}")
    if "maxLength" in schema:
        lines.append(f"{pad}- maxLength: {schema['maxLength']}")
    return "\n".join(lines)


def response_format_to_prompt_constraints(response_format: Dict[str, Any]) -> str:
    """
    把 OpenAI-style response_format(json_schema) 转成可拼到 prompt 的“硬约束段落”。
    """
    if not response_format or response_format.get("type") != "json_schema":
        return ""

    js = response_format.get("json_schema") or {}
    name = js.get("name", "schema")
    strict = js.get("strict", False)
    schema = js.get("schema") or {}

    schema_text = _schema_to_text(schema, indent=0)

    # 这段非常关键：告诉模型“只能输出 JSON、不能夹带解释、不能 markdown”
    constraints = f"""
[STRICT JSON OUTPUT CONTRACT]
You MUST output ONLY a valid JSON object (no markdown, no code fences, no extra keys, no extra text).
- Output must conform to schema name: "{name}"
- strict: {strict}

Schema:
{schema_text}

Additional rules:
- Do not wrap the JSON in ``` fences.
- Do not include any commentary before or after the JSON.
- Ensure JSON is syntactically valid (double quotes, no trailing commas).
[/STRICT JSON OUTPUT CONTRACT]
""".strip()

    return constraints


def append_schema_constraints_to_prompt(prompt: str, response_format: Optional[Dict[str, Any]]) -> str:
    """
    如果传入了 response_format，就把 schema 约束追加到 prompt 后面。
    """
    if not response_format:
        return prompt
    extra = response_format_to_prompt_constraints(response_format)
    if not extra:
        return prompt
    return prompt.rstrip() + "\n\n" + extra

    





















