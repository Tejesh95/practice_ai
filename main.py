"""
Tool Use, Parallel Tool Calls, and Partial Failure Handling — Demo
Uses Google Gemini API (Google AI Studio free tier)
"""

import os
import json
import traceback
from datetime import datetime
from dotenv import load_dotenv
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=API_KEY)


def resolve_model_name() -> str:
    """
    Model names get deprecated/shut down over time (this is exactly what
    broke gemini-1.5-flash and gemini-2.0-flash). Instead of hardcoding a
    name and hoping it's still alive, ask the API what's actually
    available RIGHT NOW and pick a Flash model that supports generateContent.
    """
    preferred_order = [
        "gemini-flash-latest",   # alias Google keeps pointed at current Flash GA
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
    ]

    available = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            available.append(m.name.replace("models/", ""))

    for candidate in preferred_order:
        if candidate in available:
            print(f"[Model resolved] Using: {candidate}")
            return candidate

    # Fall back to any flash model still alive, otherwise any model at all
    flash_models = [m for m in available if "flash" in m]
    if flash_models:
        print(f"[Model resolved] Using fallback flash model: {flash_models[0]}")
        return flash_models[0]

    if not available:
        raise RuntimeError(
            "No models supporting generateContent were returned for this API key. "
            "Check your API key / region / quota."
        )

    print(f"[Model resolved] Using fallback: {available[0]}")
    return available[0]


MODEL_NAME = resolve_model_name()

# ---------------------------------------------------------------------------
# 2. TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def get_weather(location: str) -> dict:
    """Fake weather lookup (simulated success)."""
    fake_db = {
        "hyderabad": {"temp_c": 34, "condition": "Sunny"},
        "bangalore": {"temp_c": 24, "condition": "Cloudy"},
        "mumbai": {"temp_c": 30, "condition": "Humid"},
    }
    key = location.strip().lower()
    if key not in fake_db:
        raise ValueError(f"No weather data available for '{location}'")
    return {"location": location, **fake_db[key]}


def get_stock_price(ticker: str) -> dict:
    """Simulates a broken/unreliable API — always raises (intentional)."""
    raise ConnectionError(f"Stock API timeout while fetching '{ticker}' (simulated outage)")


def calculate(expression: str) -> dict:
    """Very small, safe calculator for + - * / and parentheses only."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        raise ValueError(f"Unsafe or invalid expression: '{expression}'")
    result = eval(expression)  # safe: char-whitelisted above
    return {"expression": expression, "result": result}


TOOL_IMPLEMENTATIONS = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "calculate": calculate,
}

# ---------------------------------------------------------------------------
# 3. TOOL SCHEMAS
# ---------------------------------------------------------------------------

get_weather_decl = FunctionDeclaration(
    name="get_weather",
    description="Get current weather (temperature and condition) for a given city.",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Hyderabad'"}
        },
        "required": ["location"],
    },
)

get_stock_price_decl = FunctionDeclaration(
    name="get_stock_price",
    description="Get the current stock price for a given ticker symbol.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'GOOGLE'"}
        },
        "required": ["ticker"],
    },
)

calculate_decl = FunctionDeclaration(
    name="calculate",
    description="Evaluate a basic arithmetic expression (+, -, *, /, parentheses only).",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression, e.g. '23 * 4 + 1'"}
        },
        "required": ["expression"],
    },
)

tools = Tool(function_declarations=[get_weather_decl, get_stock_price_decl, calculate_decl])

# ---------------------------------------------------------------------------
# 4. TOOL EXECUTION LAYER
# ---------------------------------------------------------------------------

def execute_tool_call(function_call) -> dict:
    name = function_call.name
    args = dict(function_call.args) if function_call.args else {}

    record = {
        "tool": name,
        "arguments": args,
        "status": None,
        "result": None,
        "error": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if name not in TOOL_IMPLEMENTATIONS:
        record["status"] = "error"
        record["error"] = f"Unknown tool requested: '{name}'"
        return record

    try:
        result = TOOL_IMPLEMENTATIONS[name](**args)
        record["status"] = "success"
        record["result"] = result
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {str(e)}"

    return record


def execute_parallel_tool_calls(function_calls: list) -> list:
    return [execute_tool_call(fc) for fc in function_calls]

# ---------------------------------------------------------------------------
# 5. THE AGENTIC LOOP
# ---------------------------------------------------------------------------

def run_agent(user_prompt: str, max_turns: int = 5) -> dict:
    model = genai.GenerativeModel(model_name=MODEL_NAME, tools=[tools])
    chat = model.start_chat()

    transcript = {"user_prompt": user_prompt, "turns": [], "final_answer": None}

    response = chat.send_message(user_prompt)

    for turn_num in range(1, max_turns + 1):
        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_calls = [p.function_call for p in parts if p.function_call]

        turn_record = {
            "turn": turn_num,
            "function_calls_requested": len(function_calls),
            "tool_results": [],
            "model_text": None,
        }

        if not function_calls:
            text_parts = [p.text for p in parts if getattr(p, "text", None)]
            turn_record["model_text"] = "".join(text_parts)
            transcript["final_answer"] = turn_record["model_text"]
            transcript["turns"].append(turn_record)
            break

        print(f"\n[Turn {turn_num}] Model requested {len(function_calls)} tool call(s):")
        for fc in function_calls:
            print(f"   -> {fc.name}({dict(fc.args)})")

        tool_results = execute_parallel_tool_calls(function_calls)
        turn_record["tool_results"] = tool_results

        for r in tool_results:
            if r["status"] == "success":
                print(f"   [OK]    {r['tool']} -> {r['result']}")
            else:
                print(f"   [FAIL]  {r['tool']} -> {r['error']}")

        response_parts = []
        for r in tool_results:
            if r["status"] == "success":
                payload = {"status": "success", "data": r["result"]}
            else:
                payload = {"status": "error", "message": r["error"]}

            response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=r["tool"],
                        response={"result": payload},
                    )
                )
            )

        transcript["turns"].append(turn_record)
        response = chat.send_message(response_parts)

    else:
        transcript["final_answer"] = "[Max turns reached without a final answer]"

    return transcript

# ---------------------------------------------------------------------------
# 6. OUTPUT.MD GENERATOR
# ---------------------------------------------------------------------------

def write_output_md(transcript: dict, filename: str = "output.md") -> None:
    lines = []
    lines.append("# Tool Use Demo — Run Report\n")
    lines.append(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"**Model used:** `{MODEL_NAME}`\n")
    lines.append(f"**User Prompt:** `{transcript['user_prompt']}`\n")
    lines.append("---\n")

    success_count = 0
    failure_count = 0

    for turn in transcript["turns"]:
        lines.append(f"## Turn {turn['turn']}\n")

        if turn["function_calls_requested"] == 0:
            lines.append("Model responded directly with final text (no tool calls).\n")
            continue

        lines.append(f"Model requested **{turn['function_calls_requested']}** tool call(s) in parallel:\n")
        lines.append("| Tool | Arguments | Status | Result / Error |")
        lines.append("|------|-----------|--------|-----------------|")

        for r in turn["tool_results"]:
            args_str = json.dumps(r["arguments"])
            if r["status"] == "success":
                success_count += 1
                detail = f"`{json.dumps(r['result'])}`"
                status_label = "SUCCESS"
            else:
                failure_count += 1
                detail = f"`{r['error']}`"
                status_label = "FAILURE"

            lines.append(f"| `{r['tool']}` | `{args_str}` | **{status_label}** | {detail} |")

        lines.append("")

    lines.append("---\n")
    lines.append("## Final Model Answer\n")
    lines.append(transcript["final_answer"] or "(none)")
    lines.append("\n---\n")

    lines.append("## Summary: Success vs Partial Failure\n")
    lines.append(f"- **Successful tool calls:** {success_count}")
    lines.append(f"- **Failed tool calls:** {failure_count}")
    lines.append(
        "\nKey takeaway: failed tool calls were **not** hidden from the model. "
        "Each failure was reported back as an explicit `status: error` payload, "
        "which let the model reason about what it *did* have vs. what it "
        "*couldn't* retrieve, instead of silently hallucinating missing data.\n"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[output.md written to: {os.path.abspath(filename)}]")

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prompt = (
        "What's the weather in Hyderabad and Bangalore, "
        "what is the current stock price of GOOGLE, "
        "and also calculate 45 * 12 + 7?"
    )

    print("=" * 70)
    print("Running agentic tool-use loop...")
    print("=" * 70)

    try:
        transcript = run_agent(prompt)
        write_output_md(transcript)

        print("\n" + "=" * 70)
        print("FINAL ANSWER FROM MODEL:")
        print("=" * 70)
        print(transcript["final_answer"])

    except Exception:
        print("Fatal error running agent:")
        traceback.print_exc()