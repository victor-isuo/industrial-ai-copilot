"""
Vision Tool — Industrial AI Copilot Phase 3C
=============================================
Gives the LangGraph agent the ability to analyze equipment images.

Capabilities:
- Read pressure/temperature gauges and extract values
- Extract equipment nameplate data (model, ratings, serial number)
- Diagnose visible faults and damage from photos
- Analyze P&ID diagrams and identify components

Model: Gemini 2.5 Flash (multimodal)

In production this tool would accept:
- Images from plant CCTV systems
- Photos uploaded by field engineers via mobile
- Automated inspection camera feeds
- Drone imagery from equipment surveys
"""

from langchain.tools import tool
import logging
import base64
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_gemini_client():
    """Initialize Gemini client with API key."""
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env"
        )
    return genai.Client(api_key=api_key)


def _analyze_image(image_data: bytes, mime_type: str, prompt: str) -> str:
    """
    Send image to Gemini 2.5 Flash for analysis.
    
    Args:
        image_data: Raw image bytes
        mime_type: Image MIME type (image/jpeg, image/png, etc.)
        prompt: Analysis instruction
    
    Returns:
        Gemini's analysis as string
    """
    from google import genai
    from google.genai import types

    client = _get_gemini_client()

    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_data, mime_type=mime_type),
            prompt,
        ]
    )
    return result.text


@tool
def analyze_equipment_image(image_path: str, analysis_type: str = "general") -> str:
    """
    Analyze an equipment image using Gemini 2.5 Flash vision.
    Use this tool when an engineer uploads a photo of equipment,
    a gauge, a nameplate, damage, or a P&ID diagram.

    Analysis types:
        gauge — Read gauge value (pressure, temperature, flow)
        nameplate — Extract equipment nameplate data
        fault — Diagnose visible damage or fault
        pid — Analyze P&ID diagram components
        general — General equipment condition assessment

    Args:
        image_path: Path to image file OR base64-encoded image string
        analysis_type: Type of analysis to perform (default: general)
    """
    logger.info(f"Vision tool called: {analysis_type} analysis on {image_path[:50]}...")

    try:
        # Determine if input is a file path or base64 string
        if image_path.startswith("data:image"):
            # Base64 data URI — extract mime type and data
            header, data = image_path.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            image_data = base64.b64decode(data)

        elif image_path.startswith("/") or image_path.startswith("data/") or "\\" in image_path:
            # File path
            path = Path(image_path)
            if not path.exists():
                return f"Image file not found: {image_path}"

            image_data = path.read_bytes()
            suffix = path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "image/jpeg")

        else:
            # Treat as base64 string without data URI header
            try:
                image_data = base64.b64decode(image_path)
                mime_type = "image/jpeg"
            except Exception:
                return "Invalid image input. Provide a file path or base64-encoded image."

        # Build analysis prompt based on type
        prompts = {
            "gauge": """You are an industrial instrumentation expert analyzing a gauge image.
Extract the following information:
1. GAUGE TYPE: What is being measured? (pressure, temperature, flow, etc.)
2. CURRENT READING: What value does the needle/display show? Be precise.
3. SCALE RANGE: What is the full scale range shown on the gauge?
4. UNITS: What units are displayed?
5. ZONE: Is the reading in the green/normal, yellow/caution, or red/danger zone?
6. CONDITION: Is the gauge in good condition or showing any issues?

Format your response clearly with these exact headings.
If you cannot read the gauge clearly, state what is unclear.""",

            "nameplate": """You are an industrial equipment specialist analyzing an equipment nameplate.
Extract ALL visible information including:
1. MANUFACTURER: Company name
2. MODEL/TYPE: Model number or equipment type
3. SERIAL NUMBER: If visible
4. RATED POWER: kW or HP
5. RATED VOLTAGE: Volts
6. RATED CURRENT: Amps
7. RATED SPEED: RPM if shown
8. RATED PRESSURE: If applicable (psi, bar)
9. RATED FLOW: If applicable
10. EFFICIENCY CLASS: If shown
11. YEAR/DATE: Manufacturing date if visible
12. STANDARDS: Any certification marks (ISO, ATEX, etc.)

Format as a structured list. Mark any fields as 'Not visible' if unclear.""",

            "fault": """You are a senior industrial maintenance engineer analyzing equipment damage.
Provide a detailed fault assessment:
1. EQUIPMENT TYPE: What type of equipment is shown?
2. VISIBLE DAMAGE: Describe all visible damage, wear, or anomalies in detail
3. FAULT CLASSIFICATION: What type of fault is this?
   (mechanical wear, corrosion, impact damage, overheating, leakage, etc.)
4. SEVERITY: CRITICAL / WARNING / CAUTION / NORMAL
5. AFFECTED COMPONENTS: Which specific components are affected?
6. PROBABLE CAUSE: What likely caused this fault?
7. IMMEDIATE ACTION: What should be done immediately?
8. RECOMMENDED REPAIR: What maintenance or repair is needed?

Be specific and technical. This assessment will be used for maintenance planning.""",

            "pid": """You are a process engineer analyzing a P&ID (Piping and Instrumentation Diagram).
Identify and describe:
1. MAIN PROCESS EQUIPMENT: List all major equipment shown (pumps, vessels, heat exchangers, etc.)
2. PIPING: Describe the main process flow paths
3. INSTRUMENTATION: List measurement instruments shown (pressure, temperature, flow, level indicators)
4. CONTROL VALVES: Identify any control or isolation valves
5. SAFETY DEVICES: Note any relief valves, rupture disks, or safety instruments
6. PROCESS FLOW: Describe the overall process flow direction
7. TAG NUMBERS: List any visible equipment tag numbers

Focus on industrial safety and process control elements.""",

            "general": """You are an experienced industrial maintenance engineer conducting 
a general equipment condition assessment from a photo.

Provide:
1. EQUIPMENT IDENTIFICATION: What type of equipment is this?
2. OVERALL CONDITION: Good / Fair / Poor / Critical
3. VISIBLE OBSERVATIONS: Describe everything you can see about the equipment condition
4. POTENTIAL ISSUES: Any signs of wear, damage, leakage, corrosion, or abnormality?
5. MAINTENANCE RECOMMENDATIONS: What maintenance actions would you recommend?
6. SAFETY CONCERNS: Any immediate safety issues visible?

Be thorough and technical in your assessment.""",
        }

        prompt = prompts.get(analysis_type, prompts["general"])

        # Run vision analysis
        analysis = _analyze_image(image_data, mime_type, prompt)

        # Format response
        result = f"VISION ANALYSIS — {analysis_type.upper()}\n"
        result += "=" * 50 + "\n"
        result += analysis
        result += "\n" + "=" * 50
        result += "\nAnalyzed using Gemini 2.5 Flash multimodal vision."

        # Truncate if too long for agent context
        return result[:2000]

    except Exception as e:
        logger.error(f"Vision tool failed: {e}")
        return f"Vision analysis failed: {str(e)}"


@tool
def analyze_gauge_reading(image_path: str, spec_value: str = "", unit: str = "") -> str:
    """
    Read a gauge from an image and optionally check against a specification.
    Use this when an engineer uploads a photo of a pressure, temperature,
    or flow gauge and wants to know the reading or check if it is within spec.

    Args:
        image_path: Path to gauge image or base64 string
        spec_value: Optional rated spec to compare against (e.g. '380')
        unit: Unit of measurement (e.g. 'psi', 'celsius')
    """
    logger.info(f"Gauge reading tool called")

    try:
        # First get the gauge reading
        gauge_result = analyze_equipment_image.invoke({
            "image_path": image_path,
            "analysis_type": "gauge"
        })

        if not spec_value:
            return gauge_result

        # If spec provided, extract reading and run spec check
        result = gauge_result + "\n\nSPEC COMPARISON:\n"

        # Try to extract numeric reading from analysis
        import re
        numbers = re.findall(r'\b\d+\.?\d*\b', gauge_result)

        if numbers:
            # Use the most prominent number as the reading
            reading = numbers[0]
            result += f"Extracted reading: {reading} {unit}\n"
            result += f"Specification: {spec_value} {unit}\n"

            # Run spec checker
            from src.tools.spec_checker_tool import spec_checker
            spec_result = spec_checker.invoke({
                "measured_value": reading,
                "spec_value": spec_value,
                "parameter_name": "gauge reading",
                "unit": unit,
                "tolerance_percent": "10.0",
            })
            result += "\n" + spec_result
        else:
            result += "Could not extract numeric reading for spec comparison."

        return result[:2000]

    except Exception as e:
        logger.error(f"Gauge reading tool failed: {e}")
        return f"Gauge analysis failed: {str(e)}"