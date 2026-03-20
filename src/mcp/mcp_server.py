"""
MCP Server — Industrial AI Copilot
====================================
Exposes the Industrial AI Copilot's capabilities as an MCP
(Model Context Protocol) server.

Any MCP-compatible AI client — Claude Desktop, Cursor, 
custom agents — can connect to this server and access:
- Live equipment telemetry readings
- Equipment health status and alerts
- Knowledge base document search
- Spec checking against rated limits

Why MCP matters:
MCP is the emerging standard for AI-to-system integration.
Instead of building custom integrations for every AI client,
you build one MCP server and any compliant client connects
automatically. This is the production pattern for 2025+.

Running the server:
    python -m src.mcp.mcp_server

Connecting from Claude Desktop (add to claude_desktop_config.json):
    {
        "mcpServers": {
            "industrial-ai-copilot": {
                "command": "python",
                "args": ["-m", "src.mcp.mcp_server"]
            }
        }
    }
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("industrial-ai-copilot")


# ── Tool Definitions ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose available tools to MCP clients."""
    return [
        Tool(
            name="get_equipment_telemetry",
            description=(
                "Get live sensor readings for industrial equipment. "
                "Returns current operating parameters with severity classification. "
                "Available equipment: pump-001, pump-002, motor-001, compressor-001"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "equipment_id": {
                        "type": "string",
                        "description": "Equipment asset ID (e.g. 'pump-001', 'motor-001')",
                    }
                },
                "required": ["equipment_id"],
            },
        ),
        Tool(
            name="list_all_equipment",
            description=(
                "List all equipment in the plant registry with current health status. "
                "Returns overview of all assets including alert counts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter by equipment type (e.g. 'pump', 'motor')",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="check_specification",
            description=(
                "Compare a measured sensor reading against a rated specification. "
                "Returns severity classification: NORMAL, CAUTION, WARNING, or CRITICAL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "measured_value": {
                        "type": "string",
                        "description": "The actual measured value",
                    },
                    "spec_value": {
                        "type": "string",
                        "description": "The specification or rated limit",
                    },
                    "parameter_name": {
                        "type": "string",
                        "description": "What is being measured (e.g. 'discharge pressure')",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement (e.g. 'psi', 'celsius')",
                    },
                },
                "required": ["measured_value", "spec_value", "parameter_name", "unit"],
            },
        ),
        Tool(
            name="search_documentation",
            description=(
                "Search the industrial knowledge base of 27 technical documents. "
                "Returns relevant excerpts with source citations including page numbers. "
                "Use for equipment manuals, safety procedures, maintenance guides."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for",
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="convert_units",
            description=(
                "Convert between engineering units. "
                "Supports: pressure (psi, bar, kpa, mpa), "
                "temperature (celsius, fahrenheit), "
                "flow (gpm, lpm), power (kw, hp), torque (nm, lbft)"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numerical value to convert",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit (e.g. 'psi', 'celsius')",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit (e.g. 'bar', 'fahrenheit')",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        ),
    ]


# ── Tool Handlers ─────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls from MCP clients."""
    logger.info(f"MCP tool called: {name} with {arguments}")

    try:
        if name == "get_equipment_telemetry":
            result = await _handle_telemetry(arguments)

        elif name == "list_all_equipment":
            result = await _handle_list_equipment(arguments)

        elif name == "check_specification":
            result = await _handle_spec_check(arguments)

        elif name == "search_documentation":
            result = await _handle_search(arguments)

        elif name == "convert_units":
            result = await _handle_unit_conversion(arguments)

        else:
            result = f"Unknown tool: {name}"

        return CallToolResult(
            content=[TextContent(type="text", text=result)]
        )

    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool error: {str(e)}")]
        )


async def _handle_telemetry(args: dict) -> str:
    from src.api.telemetry_api import get_equipment_telemetry

    equipment_id = args.get("equipment_id", "").lower().strip()
    data = get_equipment_telemetry(equipment_id)

    if not data:
        from src.api.telemetry_api import list_equipment
        available = [e["equipment_id"] for e in list_equipment()]
        return f"Equipment '{equipment_id}' not found. Available: {', '.join(available)}"

    result = f"TELEMETRY: {data['name']} ({data['equipment_id']}) — {data['overall_health']}\n"
    result += f"Location: {data['location']} | {data['timestamp']}\n\n"
    result += "READINGS:\n"

    for param, reading in data["readings"].items():
        severity = data["severities"][param]
        flag = " ⚠" if severity in ["WARNING", "CRITICAL"] else ""
        result += (
            f" {param.replace('_', ' ').title()}: "
            f"{reading['value']} {reading['unit']} "
            f"[{severity}]{flag}\n"
        )

    if data["alerts"]:
        result += f"\nALERTS:\n"
        for alert in data["alerts"]:
            result += f" {alert['severity']}: {alert['action']}\n"

    if data.get("active_fault"):
        fault = data["active_fault"]
        result += f"\nDEVELOPING FAULT: {fault['name']} — drifting {fault['elapsed_mins']} mins\n"

    return result[:1500]


async def _handle_list_equipment(args: dict) -> str:
    from src.api.telemetry_api import list_equipment

    equipment_list = list_equipment()
    filter_str = args.get("filter", "").lower()

    if filter_str:
        equipment_list = [
            e for e in equipment_list
            if filter_str in e["equipment_id"].lower()
            or filter_str in e["name"].lower()
            or filter_str in e["type"].lower()
        ]

    if not equipment_list:
        return f"No equipment found matching '{filter_str}'."

    result = f"PLANT EQUIPMENT — {len(equipment_list)} assets\n\n"
    for eq in equipment_list:
        alert_str = f" [{eq['alert_count']} ALERT(S)]" if eq["alert_count"] > 0 else ""
        result += f" {eq['equipment_id']:<18} {eq['name']:<30} {eq['overall_health']}{alert_str}\n"

    return result


async def _handle_spec_check(args: dict) -> str:
    from src.tools.spec_checker_tool import spec_checker

    # Call the existing spec checker tool
    result = spec_checker.invoke({
        "measured_value": str(args["measured_value"]),
        "spec_value": str(args["spec_value"]),
        "parameter_name": args["parameter_name"],
        "unit": args["unit"],
        "tolerance_percent": "10.0",
    })
    return result


async def _handle_search(args: dict) -> str:
    """
    Search documentation via RAG pipeline.
    Note: Requires pipeline to be initialized.
    Returns a message directing to the HTTP API if pipeline not available.
    """
    query = args.get("query", "")

    try:
        # Try to use the global pipeline if available
        import importlib
        main_module = importlib.import_module("main")
        pipeline = getattr(main_module, "pipeline", None)

        if pipeline:
            response = pipeline.query(query)
            answer = response.answer[:800]
            result = f"ANSWER: {answer}\n\nSOURCES:\n"
            for source in list(response.sources)[:3]:
                result += f" - {source}\n"
            return result
        else:
            return (
                f"Documentation search requires the FastAPI server to be running. "
                f"Query '{query}' via POST to /query endpoint. "
                f"Or run the full server: uvicorn main:app"
            )
    except Exception as e:
        return (
            f"Documentation search unavailable in standalone MCP mode. "
            f"Use the FastAPI server at /query for full RAG search. Error: {str(e)}"
        )


async def _handle_unit_conversion(args: dict) -> str:
    from src.tools.unit_converter_tool import unit_converter

    result = unit_converter.invoke({
        "value": float(args["value"]),
        "from_unit": args["from_unit"],
        "to_unit": args["to_unit"],
    })
    return result


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    """Run the MCP server over stdio."""
    logger.info("Starting Industrial AI Copilot MCP Server...")
    logger.info("Waiting for MCP client connections...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())