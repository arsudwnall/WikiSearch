from perception import PerceptionResult
from memory import MemoryItem
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_plan(
    perception: PerceptionResult,
    memory_items: list[MemoryItem],
    tool_descriptions: str
) -> str:
    """Generate next action: tool call or final answer"""
    
    # Format memory context
    memory_context = "\n".join(f"- {m.text}" for m in memory_items) or "None"
    
    prompt = f"""
You are an AI agent that solves tasks step-by-step using tools.

Available tools:
{tool_descriptions}

Relevant context:
{memory_context}

User request:
- Input: "{perception.user_input}"
- Intent: {perception.intent}
- Entities: {perception.entities}

Instructions:
1. If you need a tool, respond: FUNCTION_CALL: tool_name|param=value
2. If you have the answer, respond: FINAL_ANSWER: [your answer]

Examples:
- FUNCTION_CALL: search_documents|query=cricket history
- FINAL_ANSWER: [Sachin Tendulkar is known as the God of Cricket]

Rules:
- Use search_documents for factual questions
- Don't repeat the same tool call
- If previous output has the answer, use FINAL_ANSWER
- Maximum 3 steps total

Your response:
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        output = response.text.strip()
        
        # Extract first valid line
        for line in output.splitlines():
            if line.startswith("FUNCTION_CALL:") or line.startswith("FINAL_ANSWER:"):
                return line.strip()
        
        return output
    
    except Exception as e:
        print(f"Decision error: {e}")
        return "FINAL_ANSWER: [error occurred]"