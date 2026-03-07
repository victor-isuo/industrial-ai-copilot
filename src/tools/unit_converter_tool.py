from langchain.tools import tool
import logging

logger = logging.getLogger(__name__)

# Conversion factors to SI base units
CONVERSIONS = {
    # Pressure (to Pascal)
    "psi_to_bar": 0.0689476,
    "bar_to_psi": 14.5038,
    "psi_to_kpa": 6.89476,
    "kpa_to_psi": 0.145038,
    "bar_to_kpa": 100,
    "kpa_to_bar": 0.01,
    "psi_to_mpa": 0.00689476,
    "mpa_to_psi": 145.038,

    # Temperature
    "celsius_to_fahrenheit": None,  # handled specially
    "fahrenheit_to_celsius": None,

    # Flow rate
    "gpm_to_lpm": 3.78541,
    "lpm_to_gpm": 0.264172,
    "m3h_to_gpm": 4.40287,
    "gpm_to_m3h": 0.227125,

    # Length
    "mm_to_inches": 0.0393701,
    "inches_to_mm": 25.4,
    "m_to_ft": 3.28084,
    "ft_to_m": 0.3048,

    # Power
    "kw_to_hp": 1.34102,
    "hp_to_kw": 0.745699,

    # Torque
    "nm_to_lbft": 0.737562,
    "lbft_to_nm": 1.35582,
}


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between engineering units commonly used in industrial settings.
    Use this tool when you need to convert measurements between unit systems.

    Supported conversions:
    - Pressure: psi, bar, kpa, mpa
    - Temperature: celsius, fahrenheit
    - Flow rate: gpm, lpm, m3h
    - Length: mm, inches, m, ft
    - Power: kw, hp
    - Torque: nm, lbft

    Args:
        value: The numerical value to convert
        from_unit: Source unit (e.g., 'psi', 'celsius', 'gpm')
        to_unit: Target unit (e.g., 'bar', 'fahrenheit', 'lpm')
    """
    logger.info(f"Unit converter: {value} {from_unit} to {to_unit}")

    try:
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()

        # Handle temperature specially
        if from_unit == "celsius" and to_unit == "fahrenheit":
            result = (value * 9/5) + 32
            return f"{value}°C = {round(result, 2)}°F"

        if from_unit == "fahrenheit" and to_unit == "celsius":
            result = (value - 32) * 5/9
            return f"{value}°F = {round(result, 2)}°C"

        # Look up conversion factor
        key = f"{from_unit}_to_{to_unit}"
        if key not in CONVERSIONS:
            available = [k for k in CONVERSIONS.keys()
                        if not k.startswith("celsius")
                        and not k.startswith("fahrenheit")]
            return (f"Conversion from {from_unit} to {to_unit} not supported. "
                   f"Available conversions: {', '.join(available)}")

        factor = CONVERSIONS[key]
        result = value * factor
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"

    except Exception as e:
        logger.error(f"Unit converter failed: {e}")
        return f"Conversion failed: {str(e)}"