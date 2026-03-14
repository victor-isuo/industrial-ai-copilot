from langchain.tools import tool
import logging

logger = logging.getLogger(__name__)

# Keywords that indicate a hard safety limit — always CRITICAL if exceeded
SAFETY_KEYWORDS = [
    "relief", "safety", "shutdown", "trip", "emergency",
    "maximum allowable", "burst", "rupture", "alarm", "cutoff"
]


@tool
def spec_checker(
    measured_value: str,
    spec_value: str,
    parameter_name: str,
    unit: str,
    tolerance_percent: str = "10.0"
) -> str:
    """
    Compare a measured reading against a documented specification.
    Use this tool when you have an actual sensor reading or measurement
    and need to check if it's within acceptable limits.

    Args:
        measured_value: The actual measured or observed value (as a number)
        spec_value: The specification or rated value from documentation (as a number)
        parameter_name: What is being measured (e.g., 'pressure', 'temperature')
        unit: The unit of measurement (e.g., 'psi', 'bar', 'rpm')
        tolerance_percent: Acceptable deviation percentage (default 10.0)
    """
    # Cast strings to float — Groq passes these as strings
    measured_value = float(measured_value)
    spec_value = float(spec_value)
    tolerance_percent = float(tolerance_percent)

    try:
        deviation = measured_value - spec_value
        deviation_percent = (deviation / spec_value) * 100
        upper_limit = spec_value * (1 + tolerance_percent / 100)
        lower_limit = spec_value * (1 - tolerance_percent / 100)
        within_spec = lower_limit <= measured_value <= upper_limit

        status = "WITHIN SPEC" if within_spec else "OUT OF SPEC"
        severity = "NORMAL"

        # Check if parameter name contains safety-critical keywords
        is_safety_limit = any(
            kw in parameter_name.lower() for kw in SAFETY_KEYWORDS
        )

        if not within_spec:
            abs_deviation = abs(deviation_percent)

            if is_safety_limit and deviation > 0:
                # Exceeding ANY safety limit is always CRITICAL
                severity = "CRITICAL"
            elif abs_deviation > 20:
                severity = "CRITICAL"
            elif abs_deviation > 10:
                severity = "WARNING"
            else:
                severity = "CAUTION"

        result = f"""
SPEC CHECK RESULT: {status}
Parameter: {parameter_name}
Measured Value: {measured_value} {unit}
Specification: {spec_value} {unit}
Acceptable Range: {round(lower_limit, 2)} - {round(upper_limit, 2)} {unit}
Deviation: {round(deviation, 2)} {unit} ({round(deviation_percent, 1)}%)
Severity: {severity}
"""

        if not within_spec:
            direction = 'above' if deviation > 0 else 'below'
            result += (
                f"\nACTION REQUIRED: {parameter_name} is "
                f"{round(abs(deviation_percent), 1)}% {direction} specification."
            )
            if severity == "CRITICAL":
                result += " IMMEDIATE shutdown and inspection required. Do not continue operation."
            elif severity == "WARNING":
                result += " Urgent inspection recommended. Monitor closely and reduce load if possible."
            elif severity == "CAUTION":
                result += " Schedule inspection at next available opportunity."

        return result.strip()

    except Exception as e:
        logger.error(f"Spec checker failed: {e}")
        return f"Spec check failed: {str(e)}"
