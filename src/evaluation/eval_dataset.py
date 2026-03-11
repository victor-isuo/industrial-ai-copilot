"""
Industrial AI Copilot — Evaluation Dataset
30 test cases across 4 categories:
- Spec checking (10 cases)
- Unit conversion (5 cases)  
- Documentation retrieval (10 cases)
- Edge cases (5 cases)
"""

EVAL_DATASET = [
    # ─── SPEC CHECKING (10 cases) ───
    {
        "id": "spec_001",
        "category": "spec_check",
        "query": "Pump pressure reading 450 psi. Spec is 380 psi. Is this dangerous?",
        "expected_tool": "spec_checker",
        "expected_severity": "WARNING",
        "expected_keywords": ["18.4", "above", "warning", "inspect"],
    },
    {
        "id": "spec_002",
        "category": "spec_check",
        "query": "Vibration reading is 2.8 mm/s. ISO spec allows 2.3 mm/s maximum.",
        "expected_tool": "spec_checker",
        "expected_severity": "WARNING",
        "expected_keywords": ["above", "spec", "vibration", "inspect"],
    },
    {
        "id": "spec_003",
        "category": "spec_check",
        "query": "Motor temperature is 95 degrees celsius. Rated maximum is 80 degrees.",
        "expected_tool": "spec_checker",
        "expected_severity": "WARNING",
        "expected_keywords": ["above", "temperature", "overheat"],
    },
    {
        "id": "spec_004",
        "category": "spec_check",
        "query": "Flow rate measured at 120 lpm. Design spec is 150 lpm.",
        "expected_tool": "spec_checker",
        "expected_severity": "WITHIN SPEC",
        "expected_keywords": ["below", "flow", "check"],
    },
    {
        "id": "spec_005",
        "category": "spec_check",
        "query": "Bearing temperature 65 celsius. Max rated temperature is 80 celsius.",
        "expected_tool": "spec_checker",
        "expected_severity": "WITHIN SPEC",
        "expected_keywords": ["within", "normal", "acceptable"],
    },
    {
        "id": "spec_006",
        "category": "spec_check",
        "query": "Discharge pressure 600 psi. Safety relief valve set at 500 psi.",
        "expected_tool": "spec_checker",
        "expected_severity": "CRITICAL",
        "expected_keywords": ["critical", "dangerous", "immediate", "shutdown"],
    },
    {
        "id": "spec_007",
        "category": "spec_check",
        "query": "Shaft speed 3200 rpm. Rated speed is 3000 rpm maximum.",
        "expected_tool": "spec_checker",
        "expected_severity": "CAUTION",
        "expected_keywords": ["above", "speed", "check"],
    },
    {
        "id": "spec_008",
        "category": "spec_check",
        "query": "Oil pressure 18 psi. Minimum required pressure is 25 psi.",
        "expected_tool": "spec_checker",
        "expected_severity": "WARNING",
        "expected_keywords": ["below", "low", "oil", "pressure"],
    },
    {
        "id": "spec_009",
        "category": "spec_check",
        "query": "Current draw 45 amps. Motor rated at 40 amps full load.",
        "expected_tool": "spec_checker",
        "expected_severity": "CAUTION",
        "expected_keywords": ["above", "current", "motor"],
    },
    {
        "id": "spec_010",
        "category": "spec_check",
        "query": "Tank level at 85%. High level alarm set at 90%.",
        "expected_tool": "spec_checker",
        "expected_severity": "WITHIN SPEC",
        "expected_keywords": ["within", "normal", "monitor"],
    },

    # ─── UNIT CONVERSION (5 cases) ───
    {
        "id": "unit_001",
        "category": "unit_conversion",
        "query": "Convert 150 psi to bar",
        "expected_tool": "unit_converter",
        "expected_keywords": ["10.34", "bar"],
    },
    {
        "id": "unit_002",
        "category": "unit_conversion",
        "query": "Convert 100 degrees celsius to fahrenheit",
        "expected_tool": "unit_converter",
        "expected_keywords": ["212", "fahrenheit"],
    },
    {
        "id": "unit_003",
        "category": "unit_conversion",
        "query": "Convert 50 gallons per minute to liters per minute",
        "expected_tool": "unit_converter",
        "expected_keywords": ["189", "lpm", "liters"],
    },
    {
        "id": "unit_004",
        "category": "unit_conversion",
        "query": "Convert 75 kilowatts to horsepower",
        "expected_tool": "unit_converter",
        "expected_keywords": ["100", "horsepower", "hp"],
    },
    {
        "id": "unit_005",
        "category": "unit_conversion",
        "query": "Convert 500 psi to MPa",
        "expected_tool": "unit_converter",
        "expected_keywords": ["3.44", "mpa"],
    },

    # ─── DOCUMENTATION RETRIEVAL (10 cases) ───
    {
        "id": "doc_001",
        "category": "retrieval",
        "query": "What should I do if a gear pump loses suction?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["suction", "pump", "check", "inspect"],
    },
    {
        "id": "doc_002",
        "category": "retrieval",
        "query": "What PPE is required when handling petroleum products?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["gloves", "protective", "PPE", "safety"],
    },
    {
        "id": "doc_003",
        "category": "retrieval",
        "query": "What are the ISO 9001 quality management requirements?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["ISO", "quality", "management", "requirements"],
    },
    {
        "id": "doc_004",
        "category": "retrieval",
        "query": "What are the lockout tagout procedures?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["lockout", "tagout", "energy", "isolation"],
    },
    {
        "id": "doc_005",
        "category": "retrieval",
        "query": "How do I perform predictive maintenance on rotating equipment?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["predictive", "maintenance", "vibration", "monitoring"],
    },
    {
        "id": "doc_006",
        "category": "retrieval",
        "query": "What are the confined space entry requirements?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["confined", "space", "entry", "permit"],
    },
    {
        "id": "doc_007",
        "category": "retrieval",
        "query": "What causes bearing failure in centrifugal pumps?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["bearing", "failure", "lubrication", "wear"],
    },
    {
        "id": "doc_008",
        "category": "retrieval",
        "query": "What are the safety requirements for hydraulic systems?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["hydraulic", "safety", "pressure", "relief"],
    },
    {
        "id": "doc_009",
        "category": "retrieval",
        "query": "How do I read a safety data sheet?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["SDS", "hazard", "section", "chemical"],
    },
    {
        "id": "doc_010",
        "category": "retrieval",
        "query": "What are the ISO 45001 occupational health requirements?",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["ISO", "45001", "health", "safety"],
    },

    # ─── EDGE CASES (5 cases) ───
    {
        "id": "edge_001",
        "category": "edge_case",
        "query": "The equipment is making a strange noise",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["noise", "vibration", "inspect", "check"],
    },
    {
        "id": "edge_002",
        "category": "edge_case",
        "query": "Convert psi to happiness",
        "expected_tool": "unit_converter",
        "expected_keywords": ["not supported", "invalid", "cannot"],
    },
    {
        "id": "edge_003",
        "category": "edge_case",
        "query": "Pump is on fire",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["emergency", "shutdown", "fire", "safety"],
    },
    {
        "id": "edge_004",
        "category": "edge_case",
        "query": "What is the meaning of life?",
        "expected_tool": None,
        "expected_keywords": ["cannot", "outside", "scope", "engineering"],
    },
    {
        "id": "edge_005",
        "category": "edge_case",
        "query": "Pressure reading 450 psi but I don't know what the spec is",
        "expected_tool": "search_industrial_documentation",
        "expected_keywords": ["spec", "manual", "documentation", "refer"],
    },
]

if __name__ == "__main__":
    print(f"Evaluation dataset loaded: {len(EVAL_DATASET)} test cases")
    categories = {}
    for case in EVAL_DATASET:
        cat = case["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in categories.items():
        print(f"  {cat}: {count} cases")