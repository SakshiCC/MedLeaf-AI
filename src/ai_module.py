from groq import Groq
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_ai_response(plant_info, user_query):

    prompt = f"""
You are an expert in medicinal plants.

Use ONLY the plant data provided.

Plant Data:
{plant_info}

User Question:
{user_query}

Respond in this format:

🌿 Plant Name:
💊 Disease Targeted:
🧪 Preparation Method:
⏳ Usage Instructions:
⚠️ Precautions:
📚 Scientific Reasoning:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"