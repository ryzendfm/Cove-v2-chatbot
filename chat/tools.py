from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Returns the current system date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Safely evaluates a basic mathematical expression. Example: '45 * 12 + 100'"""
    try:
        # Allow only safe math characters
        allowed = set("0123456789+-*/(). %")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in mathematical expression."
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# List of all available tools
AVAILABLE_TOOLS = [
    get_current_time,
    calculate,
]

# Map tool name -> tool function for easy invocation
TOOL_MAP = {t.name: t for t in AVAILABLE_TOOLS}


def execute_tool_call(tool_call: dict) -> str:
    """
    Executes a single tool call dictionary produced by LangChain/Groq
    and returns the result string.
    """
    name = tool_call.get("name")
    args = tool_call.get("args", {})

    if name in TOOL_MAP:
        selected_tool = TOOL_MAP[name]
        try:
            output = selected_tool.invoke(args)
            return str(output)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"
    else:
        return f"Tool '{name}' is not recognized."
