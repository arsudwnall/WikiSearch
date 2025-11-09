from pydantic import BaseModel
from google import genai
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class PerceptionResult(BaseModel):
    """Structured understanding of user input"""
    user_input: str
    intent: str
    entities: list[str] = []
    tool_hint: Optional[str] = None

def extract_perception(user_input: str) -> PerceptionResult:
    """Extract intent, entities, and tool suggestions from user input"""
    
    prompt = f"""
Extract structured information from this user input: "{user_input}"

Return a Python dict with:
- intent: brief description of what user wants
- entities: list of key terms/values
- tool_hint: suggested tool name (if applicable)

Output only the dict, no formatting.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Parse LLM response
        raw = response.text.strip().replace("```json", "").replace("```", "")
        parsed = eval(raw)
        
        # Handle entities - convert to list of strings
        entities = parsed.get("entities", [])
        if isinstance(entities, dict):
            # If entities is a dict, convert to list of values
            parsed["entities"] = list(entities.values())
        elif isinstance(entities, list):
            # If entities is a list, ensure all items are strings
            entity_strings = []
            for item in entities:
                if isinstance(item, dict):
                    # Extract value from dict (prefer 'value' key, then 'entity' key, then any value)
                    if 'value' in item:
                        entity_strings.append(str(item['value']))
                    elif 'entity' in item:
                        entity_strings.append(str(item['entity']))
                    else:
                        # Just take the first value from the dict
                        entity_strings.append(str(list(item.values())[0]))
                elif isinstance(item, str):
                    entity_strings.append(item)
                else:
                    entity_strings.append(str(item))
            parsed["entities"] = entity_strings
        else:
            parsed["entities"] = []
        
        return PerceptionResult(user_input=user_input, **parsed)
    
    except Exception as e:
        print(f"Perception error: {e}")
        return PerceptionResult(
            user_input=user_input,
            intent="unknown",
            entities=[]
        )