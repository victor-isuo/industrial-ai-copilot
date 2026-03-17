"""
Telemetry API — Industrial AI Copilot
======================================
Simulates realistic equipment sensor readings with:
- Time-based drift — readings change gradually over time
- Fault injection — developing faults that worsen over time
- Equipment-specific parameters — each machine type has correct ranges
- Multi-equipment support — query any asset by equipment_id

In production this module would be replaced by calls to:
- A plant historian API (OSIsoft PI, InfluxDB)
- An MQTT broker subscription
- A SCADA system REST API
- An IoT platform like AWS IoT, Azure IoT Hub, or Siemens MindSphere

The agent tool interface remains identical regardless of data source.
"""

import math
import random
import time
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Equipment Registry ────────────────────────────────────────────────────────
# Each equipment type defines normal operating ranges and fault thresholds.
# Values are based on realistic industrial specifications.

EQUIPMENT_REGISTRY = {
    "pump-001": {
        "name":        "Gear Pump Unit 1",
        "type":        "gear_pump",
        "location":    "Pump House A",
        "installed":   "2019-03-15",
        "parameters": {
            "discharge_pressure": {"unit": "psi",  "normal_min": 340, "normal_max": 420, "warning": 450, "critical": 500,  "base": 380},
            "suction_pressure":   {"unit": "psi",  "normal_min": 10,  "normal_max": 30,  "warning": 8,   "critical": 5,   "base": 20},
            "flow_rate":          {"unit": "lpm",  "normal_min": 130, "normal_max": 170, "warning": 120, "critical": 100, "base": 150},
            "temperature":        {"unit": "°C",   "normal_min": 40,  "normal_max": 75,  "warning": 85,  "critical": 95,  "base": 60},
            "vibration":          {"unit": "mm/s", "normal_min": 0.5, "normal_max": 2.3, "warning": 2.8, "critical": 4.5, "base": 1.2},
            "shaft_speed":        {"unit": "RPM",  "normal_min": 1400,"normal_max": 1550,"warning": 1600,"critical": 1700,"base": 1480},
        }
    },
    "pump-002": {
        "name":        "Centrifugal Pump Unit 2",
        "type":        "centrifugal_pump",
        "location":    "Pump House B",
        "installed":   "2020-07-22",
        "parameters": {
            "discharge_pressure": {"unit": "psi",  "normal_min": 80,  "normal_max": 120, "warning": 135, "critical": 150, "base": 100},
            "suction_pressure":   {"unit": "psi",  "normal_min": 5,   "normal_max": 20,  "warning": 3,   "critical": 1,   "base": 12},
            "flow_rate":          {"unit": "lpm",  "normal_min": 400, "normal_max": 600, "warning": 350, "critical": 280, "base": 500},
            "temperature":        {"unit": "°C",   "normal_min": 35,  "normal_max": 65,  "warning": 75,  "critical": 90,  "base": 50},
            "vibration":          {"unit": "mm/s", "normal_min": 0.3, "normal_max": 2.0, "warning": 2.5, "critical": 4.0, "base": 1.0},
            "shaft_speed":        {"unit": "RPM",  "normal_min": 2800,"normal_max": 3000,"warning": 3100,"critical": 3300,"base": 2950},
        }
    },
    "motor-001": {
        "name":        "Drive Motor Unit 1",
        "type":        "electric_motor",
        "location":    "Motor Control Room A",
        "installed":   "2018-11-10",
        "parameters": {
            "winding_temperature": {"unit": "°C",   "normal_min": 40,  "normal_max": 80,  "warning": 90,  "critical": 105, "base": 65},
            "bearing_temperature": {"unit": "°C",   "normal_min": 35,  "normal_max": 70,  "warning": 80,  "critical": 95,  "base": 55},
            "current_draw":        {"unit": "A",    "normal_min": 30,  "normal_max": 42,  "warning": 45,  "critical": 50,  "base": 38},
            "vibration":           {"unit": "mm/s", "normal_min": 0.2, "normal_max": 1.8, "warning": 2.3, "critical": 3.5, "base": 0.9},
            "shaft_speed":         {"unit": "RPM",  "normal_min": 1450,"normal_max": 1500,"warning": 1520,"critical": 1550,"base": 1480},
            "insulation_resistance":{"unit": "MΩ",  "normal_min": 100, "normal_max": 999, "warning": 50,  "critical": 10,  "base": 500},
        }
    },
    "compressor-001": {
        "name":        "Air Compressor Unit 1",
        "type":        "reciprocating_compressor",
        "location":    "Utility Room",
        "installed":   "2021-02-28",
        "parameters": {
            "discharge_pressure": {"unit": "psi",  "normal_min": 100, "normal_max": 145, "warning": 150, "critical": 165, "base": 120},
            "suction_pressure":   {"unit": "psi",  "normal_min": 12,  "normal_max": 18,  "warning": 10,  "critical": 8,   "base": 14},
            "discharge_temp":     {"unit": "°C",   "normal_min": 100, "normal_max": 150, "warning": 165, "critical": 180, "base": 130},
            "oil_pressure":       {"unit": "psi",  "normal_min": 25,  "normal_max": 45,  "warning": 20,  "critical": 15,  "base": 35},
            "vibration":          {"unit": "mm/s", "normal_min": 1.0, "normal_max": 3.5, "warning": 4.5, "critical": 7.0, "base": 2.0},
            "rpm":                {"unit": "RPM",  "normal_min": 900, "normal_max": 1050,"warning": 1100,"critical": 1200,"base": 980},
        }
    },
}

# ── Fault Scenarios ───────────────────────────────────────────────────────────
# Each fault scenario defines which parameter drifts and how fast.
# severity 0.0 = fault just starting, 1.0 = fully developed fault

FAULT_SCENARIOS = {
    "pump-001": {
        "name":      "Developing Bearing Wear",
        "parameter": "vibration",
        "direction": "up",      # reading drifts upward
        "rate":      0.15,      # units per minute of simulated time
        "active":    True,
    },
    "pump-002": {
        "name":      "Suction Cavitation",
        "parameter": "suction_pressure",
        "direction": "down",
        "rate":      0.8,
        "active":    True,
    },
    "motor-001": {
        "name":      "Bearing Overheating",
        "parameter": "bearing_temperature",
        "direction": "up",
        "rate":      0.5,
        "active":    True,
    },
    "compressor-001": {
        "name":      "Oil Pressure Degradation",
        "parameter": "oil_pressure",
        "direction": "down",
        "rate":      0.3,
        "active":    True,
    },
}

# Track when simulation started for drift calculations
_sim_start_time = time.time()


def _get_sim_minutes() -> float:
    """Minutes elapsed since simulation started — drives drift."""
    return (time.time() - _sim_start_time) / 60.0


def _get_reading(equipment_id: str, param_name: str, param_config: dict) -> float:
    """
    Generate a realistic sensor reading with:
    - Small random noise around base value
    - Gradual drift over time if a fault scenario is active
    """
    base  = param_config["base"]
    noise = (param_config["normal_max"] - param_config["normal_min"]) * 0.02
    value = base + random.uniform(-noise, noise)

    # Apply fault drift if active for this equipment + parameter
    fault = FAULT_SCENARIOS.get(equipment_id)
    if fault and fault["active"] and fault["parameter"] == param_name:
        minutes = _get_sim_minutes()
        drift   = fault["rate"] * minutes

        if fault["direction"] == "up":
            value += drift
        else:
            value -= drift

    return round(value, 2)


def _classify_severity(value: float, param_config: dict, param_name: str) -> dict:
    """
    Classify reading severity against warning and critical thresholds.
    Handles both high-side and low-side limits correctly.
    """
    normal_min = param_config["normal_min"]
    normal_max = param_config["normal_max"]
    warning    = param_config["warning"]
    critical   = param_config["critical"]
    unit       = param_config["unit"]

    # Determine if this parameter fails high or low
    # If warning > normal_max → fails high (pressure, temperature, vibration)
    # If warning < normal_min → fails low (suction pressure, flow, oil pressure)
    fails_high = warning > normal_max

    if fails_high:
        if value >= critical:
            severity = "CRITICAL"
            action   = f"{param_name} is critically high at {value} {unit}. Immediate shutdown required."
        elif value >= warning:
            severity = "WARNING"
            action   = f"{param_name} is above warning threshold at {value} {unit}. Urgent inspection needed."
        elif value > normal_max:
            severity = "CAUTION"
            action   = f"{param_name} is slightly above normal range at {value} {unit}. Monitor closely."
        elif value < normal_min:
            severity = "CAUTION"
            action   = f"{param_name} is below normal range at {value} {unit}. Check operating conditions."
        else:
            severity = "NORMAL"
            action   = f"{param_name} is within normal operating range."
    else:
        # Fails low
        if value <= critical:
            severity = "CRITICAL"
            action   = f"{param_name} is critically low at {value} {unit}. Immediate shutdown required."
        elif value <= warning:
            severity = "WARNING"
            action   = f"{param_name} is below warning threshold at {value} {unit}. Urgent inspection needed."
        elif value < normal_min:
            severity = "CAUTION"
            action   = f"{param_name} is below normal range at {value} {unit}. Monitor closely."
        else:
            severity = "NORMAL"
            action   = f"{param_name} is within normal operating range."

    return {
        "severity": severity,
        "action":   action,
    }


def get_equipment_telemetry(equipment_id: str) -> Optional[dict]:
    """
    Get current telemetry readings for a specific equipment asset.
    Returns None if equipment_id not found.
    """
    equipment = EQUIPMENT_REGISTRY.get(equipment_id)
    if not equipment:
        return None

    readings   = {}
    severities = {}
    alerts     = []

    for param_name, param_config in equipment["parameters"].items():
        value    = _get_reading(equipment_id, param_name, param_config)
        classify = _classify_severity(value, param_config, param_name)

        readings[param_name] = {
            "value":    value,
            "unit":     param_config["unit"],
            "normal":   f"{param_config['normal_min']} - {param_config['normal_max']} {param_config['unit']}",
        }

        severities[param_name] = classify["severity"]

        if classify["severity"] in ["WARNING", "CRITICAL"]:
            alerts.append({
                "parameter": param_name,
                "severity":  classify["severity"],
                "action":    classify["action"],
            })

    # Overall health — worst severity across all parameters
    severity_rank = {"NORMAL": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}
    overall = max(severities.values(), key=lambda s: severity_rank.get(s, 0))

    # Active fault scenario info
    fault = FAULT_SCENARIOS.get(equipment_id)
    fault_info = None
    if fault and fault["active"]:
        minutes   = _get_sim_minutes()
        fault_info = {
            "name":         fault["name"],
            "parameter":    fault["parameter"],
            "elapsed_mins": round(minutes, 1),
        }

    return {
        "equipment_id":   equipment_id,
        "name":           equipment["name"],
        "type":           equipment["type"],
        "location":       equipment["location"],
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "overall_health": overall,
        "readings":       readings,
        "severities":     severities,
        "alerts":         alerts,
        "active_fault":   fault_info,
    }


def list_equipment() -> list:
    """Return summary of all equipment in the registry."""
    result = []
    for eq_id, eq_data in EQUIPMENT_REGISTRY.items():
        telemetry = get_equipment_telemetry(eq_id)
        result.append({
            "equipment_id":   eq_id,
            "name":           eq_data["name"],
            "type":           eq_data["type"],
            "location":       eq_data["location"],
            "overall_health": telemetry["overall_health"] if telemetry else "UNKNOWN",
            "alert_count":    len(telemetry["alerts"]) if telemetry else 0,
        })
    return result


def reset_simulation():
    """Reset drift timer — useful for demos."""
    global _sim_start_time
    _sim_start_time = time.time()
    logger.info("Telemetry simulation reset")