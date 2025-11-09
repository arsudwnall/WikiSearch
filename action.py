from pydantic import BaseModel
from mcp import ClientSession
import ast

class ToolCallResult(BaseModel):
    """Result of a tool execution"""
    tool_name: str
    arguments: dict
    result: str | list | dict

def parse_function_call(response: str) -> tuple[str, dict]:
    """Parse FUNCTION_CALL string into tool name and arguments"""
    
    if not response.startswith("FUNCTION_CALL:"):
        raise ValueError("Not a valid FUNCTION_CALL")
    
    # Extract function info: "FUNCTION_CALL: tool_name|param1=value1|param2=value2"
    _, function_info = response.split(":", 1)
    parts = [p.strip() for p in function_info.split("|")]
    
    tool_name = parts[0]
    arguments = {}
    
    # Parse parameters
    for part in parts[1:]:
        if "=" not in part:
            continue
        
        key, value = part.split("=", 1)
        
        # Try to parse as Python literal
        try:
            parsed_value = ast.literal_eval(value)
        except:
            parsed_value = value.strip()
        
        # Handle nested keys (e.g., "input.string")
        keys = key.split(".")
        current = arguments
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = parsed_value
    
    return tool_name, arguments

async def execute_tool(
    session: ClientSession,
    tools: list,
    response: str
) -> ToolCallResult:
    """Execute a tool via MCP"""
    
    # Parse the function call
    tool_name, arguments = parse_function_call(response)
    
    # Verify tool exists
    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")
    
    print(f"Calling {tool_name} with {arguments}")
    
    # Execute via MCP
    result = await session.call_tool(tool_name, arguments=arguments)
    
    # Extract text result
    if hasattr(result, 'content'):
        if isinstance(result.content, list):
            output = [getattr(item, 'text', str(item)) for item in result.content]
        else:
            output = getattr(result.content, 'text', str(result.content))
    else:
        output = str(result)
    
    return ToolCallResult(
        tool_name=tool_name,
        arguments=arguments,
        result=output
    )