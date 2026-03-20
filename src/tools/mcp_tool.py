"""
MCP Client Tool — Industrial AI Copilot
=========================================
Gives the LangGraph agent the ability to connect to and query
any MCP-compatible server as a client.

This demonstrates the full MCP loop:
- Your system IS an MCP server (mcp_server.py)
- Your agent also CONSUMES MCP servers (this file)

In production this would connect to:
- External plant systems exposed as MCP servers
- Third-party industrial data providers
- Other AI systems in a multi-agent MCP network

Why this matters:
MCP creates a standard interface between AI agents and tools.
Instead of writing custom integrations for every data source,
any MCP-compatible system can be connected with minimal code.
This is the production pattern for enterprise AI in 2025+.
"""

from langchain.tools import tool
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async coroutine from sync context safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except Exception as e:
        raise e


async def _query_mcp_server(server_command: list, tool_name: str, tool_args: dict) -> str:
    """
    Connect to an MCP server via stdio and call a tool.
    
    Args:
        server_command: Command to launch the MCP server
        tool_name:      Name of the tool to call
        tool_args:      Arguments to pass to the tool
    
    Returns:
        Tool result as string
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=server_command[0],
            args=server_command[1:],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools for logging
                tools_result = await session.list_tools()
                tool_names   = [t.name for t in tools_result.tools]
                logger.info(f"MCP server tools available: {tool_names}")

                # Call the requested tool
                result = await session.call_tool(tool_name, tool_args)

                # Extract text content from result
                if result.content:
                    return "\n".join(
                        item.text for item in result.content
                        if hasattr(item, "text")
                    )
                return "No result returned from MCP server."

    except Exception as e:
        logger.error(f"MCP client error: {e}")
        raise


@tool
def query_mcp_industrial_server(tool_name: str, equipment_id: str = "") -> str:
    """
    Query the Industrial AI Copilot MCP server directly.
    Use this to demonstrate MCP protocol integration — connecting to
    the industrial system as an external MCP client would.

    Available tool_name values:
        get_equipment_telemetry  — requires equipment_id
        list_all_equipment       — no equipment_id needed
    
    Args:
        tool_name:    MCP tool to call on the industrial server
        equipment_id: Equipment ID for telemetry queries (e.g. 'pump-001')
    """
    logger.info(f"MCP client tool called: {tool_name} for {equipment_id}")

    try:
        import sys
        import os

        # Build server command — launches our MCP server as a subprocess
        server_command = [sys.executable, "-m", "src.mcp.mcp_server"]

        # Build tool arguments
        if tool_name == "get_equipment_telemetry":
            if not equipment_id:
                return "equipment_id is required for get_equipment_telemetry"
            tool_args = {"equipment_id": equipment_id}

        elif tool_name == "list_all_equipment":
            tool_args = {}

        else:
            return (
                f"Unknown MCP tool: {tool_name}. "
                f"Available: get_equipment_telemetry, list_all_equipment"
            )

        # Run async MCP client call
        result = _run_async(
            _query_mcp_server(server_command, tool_name, tool_args)
        )

        return f"[MCP] {result}"

    except Exception as e:
        logger.error(f"MCP tool failed: {e}")
        # Graceful fallback — use direct telemetry if MCP fails
        logger.info("Falling back to direct telemetry...")
        try:
            from src.api.telemetry_api import get_equipment_telemetry, list_equipment

            if tool_name == "get_equipment_telemetry" and equipment_id:
                data = get_equipment_telemetry(equipment_id)
                if data:
                    return (
                        f"[Direct] {data['name']} — {data['overall_health']} | "
                        f"Alerts: {len(data['alerts'])}"
                    )
            elif tool_name == "list_all_equipment":
                equipment = list_equipment()
                return "[Direct] " + ", ".join(
                    f"{e['equipment_id']}:{e['overall_health']}"
                    for e in equipment
                )
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")

        return f"MCP connection failed: {str(e)}"

