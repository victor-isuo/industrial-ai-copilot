from langchain.tools import tool
import math
import logging

logger = logging.getLogger(__name__)


@tool
def engineering_calculator(expression: str) -> str:
    """
    Perform engineering calculations safely.
    Use this tool for mathematical computations involving:
    - Pressure calculations
    - Flow rate computations  
    - Temperature conversions
    - Power and torque calculations
    - Percentage differences and deviations
    - Any arithmetic needed for engineering analysis

    Args:
        expression: Mathematical expression to evaluate.
        Examples: "15 * 1.1", "(450 - 380) / 380 * 100", "math.pi * 0.05**2"
    """
    logger.info(f"Calculator tool called with: {expression}")

    try:
        # Safe evaluation with math functions available
        allowed_names = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
        }

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"

    except Exception as e:
        logger.error(f"Calculator failed: {e}")
        return f"Calculation failed: {str(e)}. Please check the expression syntax."