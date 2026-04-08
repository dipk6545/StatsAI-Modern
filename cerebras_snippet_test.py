from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import os

# 1. Load keys from your backend server config
load_dotenv('server/.env')
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY")

# 2. Initialize the official Cerebras Cloud Client
client = Cerebras(api_key=CEREBRAS_KEY)

# 3. Create the chat completion using the 2026 Heavyweight model
chat_completion = client.chat.completions.create(
    model="qwen3-235b-instruct",
    messages=[
        {
            "role": "user", 
            "content": "Analyze yourself: Are you the Qwen-235B model? Also generate a raw <data_manifest> for a sample Normal Plot."
        }
    ],
    max_tokens=8192
)

# 4. Print the raw response from the Cerebras LPU Endpoint
print("-" * 60)
print("🤖 CEREBRAS LPU DIRECT OUTPUT (235B):")
print("-" * 60)
print(chat_completion.choices[0].message.content)
print("-" * 60)
