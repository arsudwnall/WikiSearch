import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from perception import extract_perception
from memory import MemoryManager
from decision import generate_plan
from action import execute_tool

MAX_STEPS = 3

async def main(user_query: str):
    """Main agent loop"""
    
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        cwd="D:/Projects/WikiSearch_Simplified/code"
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools
            tools_result = await session.list_tools()
            tools = tools_result.tools
            tool_descriptions = "\n".join(
                f"- {t.name}: {t.description}" for t in tools
            )
            
            print(f"Loaded {len(tools)} tools")
            
            # Initialize memory and tracking
            memory = MemoryManager()
            original_query = user_query
            
            # Agent loop
            for step in range(MAX_STEPS):
                print(f"\n--- Step {step + 1} ---")
                
                # Understand user input
                perception = extract_perception(user_query)
                print(f"Intent: {perception.intent}")
                
                # Retrieve relevant memories
                relevant_memories = memory.retrieve(user_query, top_k=3)
                
                # Generate next action
                plan = generate_plan(perception, relevant_memories, tool_descriptions)
                print(f"Plan: {plan}")
                
                # Check if done
                if plan.startswith("FINAL_ANSWER:"):
                    answer = plan.replace("FINAL_ANSWER:", "").strip()
                    print(f"\n✅ Final Answer: {answer}")
                    break
                
                # Execute tool
                result = await execute_tool(session, tools, plan)
                print(f"Tool result: {result.result}")
                
                # Store in memory
                memory.add_tool_result(
                    tool_name=result.tool_name,
                    arguments=result.arguments,
                    result=result.result,
                    user_query=user_query
                )
                
                # Update query with context
                user_query = f"Original: {original_query}\nPrevious: {result.result}\nNext step?"
            
            print("\n✓ Agent session complete")

if __name__ == "__main__":
    query = input("What would you like to know? → ")
    asyncio.run(main(query))