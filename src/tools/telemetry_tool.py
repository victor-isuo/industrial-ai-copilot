"""
Telemetry Tool — Industrial AI Copilot
=======================================
LangGraph tool that gives the agent access to live equipment telemetry.

The agent calls this tool when:
- An engineer asks about current equipment readings
- A query mentions specific equipment by ID or name
- Diagnosing a fault requires knowing current sensor values
- Combining live data with documentation for full diagnosis

In production this tool would call a real plant API, MQTT broker,
or historian database. The interface is identical.
"""

from langchain.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def get_equipment_telemetry(equipment_id: str) -> str:
    """
    Retrieve live sensor readings for industrial equipment.
    Use this tool when you need current operating data for a specific machine.
    Call this before spec_checker when diagnosing equipment issues —
    it provides the actual measured values to check against specifications.

    Available equipment IDs:
        pump-001        — Gear Pump Unit 1 (Pump House A)
        pump-002        — Centrifugal Pump Unit 2 (Pump House B)
        motor-001       — Drive Motor Unit 1 (Motor Control Room A)
        compressor-001  — Air Compressor Unit 1 (Utility Room)

    Args:
        equipment_id: Equipment asset ID (e.g. 'pump-001', 'motor-001')

    Returns:
        Current sensor readings with severity classification and alerts
    """
    logger.info(f"Telemetry tool called for: {equipment_id}")

    try:
        from src.api.telemetry_api import get_equipment_telemetry as fetch_telemetry

        data = fetch_telemetry(equipment_id.lower().strip())

        if data is None:
            from src.api.telemetry_api import list_equipment
            available = [e["equipment_id"] for e in list_equipment()]
            return (
                f"Equipment '{equipment_id}' not found in registry. "
                f"Available equipment IDs: {', '.join(available)}"
            )

        # Format response for the agent
        result  = f"TELEMETRY REPORT — {data['name']} ({data['equipment_id']})\n"
        result += f"Location: {data['location']}\n"
        result += f"Timestamp: {data['timestamp']}\n"
        result += f"Overall Health: {data['overall_health']}\n\n"

        result += "SENSOR READINGS:\n"
        for param, reading in data["readings"].items():
            severity = data["severities"][param]
            result += (
                f"  {param.replace('_', ' ').title()}: "
                f"{reading['value']} {reading['unit']} "
                f"[Normal: {reading['normal']}] — {severity}\n"
            )

        if data["alerts"]:
            result += f"\nACTIVE ALERTS ({len(data['alerts'])}):\n"
            for alert in data["alerts"]:
                result += f"  ⚠ {alert['severity']} — {alert['action']}\n"
        else:
            result += "\nNo active alerts. All parameters within normal range.\n"

        if data.get("active_fault"):
            fault = data["active_fault"]
            result += (
                f"\nDEVELOPING FAULT DETECTED:\n"
                f"  {fault['name']} — {fault['parameter'].replace('_', ' ')} "
                f"has been drifting for {fault['elapsed_mins']} minutes.\n"
                f"  Recommend immediate inspection and cross-reference with maintenance documentation.\n"
            )

        return result

    except Exception as e:
        logger.error(f"Telemetry tool failed: {e}")
        return f"Telemetry retrieval failed: {str(e)}"


@tool
def list_all_equipment(query: str = "") -> str:
    """
    List all equipment in the plant registry with current health status.
    Use this tool when an engineer asks about overall plant status,
    wants to know which equipment is available, or needs a health overview.

    Args:
        query: Optional filter string (e.g. 'pump', 'motor') — leave empty for all equipment
    """
    logger.info(f"List equipment tool called, filter: '{query}'")

    try:
        from src.api.telemetry_api import list_equipment

        equipment_list = list_equipment()

        if query:
            equipment_list = [
                e for e in equipment_list
                if query.lower() in e["equipment_id"].lower()
                or query.lower() in e["name"].lower()
                or query.lower() in e["type"].lower()
            ]

        if not equipment_list:
            return f"No equipment found matching '{query}'."

        result  = f"PLANT EQUIPMENT REGISTRY — {len(equipment_list)} assets\n\n"

        for eq in equipment_list:
            alert_str = f" — {eq['alert_count']} ALERT(S)" if eq["alert_count"] > 0 else ""
            result += (
                f"  {eq['equipment_id']:<18} {eq['name']:<35} "
                f"Health: {eq['overall_health']}{alert_str}\n"
            )

        result += "\nUse get_equipment_telemetry(equipment_id) for detailed readings on any asset."
        return result

    except Exception as e:
        logger.error(f"List equipment failed: {e}")
        return f"Failed to retrieve equipment list: {str(e)}"