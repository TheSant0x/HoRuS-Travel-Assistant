from huggingface_hub import InferenceClient
import matplotlib.pyplot as plt
from pprint import pprint
import os

client = InferenceClient(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    provider="auto",   # Automatically selects best provider
)

models = [
    "google/gemma-2-2b-it",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-R1"
]

def call_model(model, prompt):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}],
)
    return response, response.choices[0].message.get("content")

def extract_hfmodel_name(model):
    parts = model.split("/")
    company = parts[0]
    model_name = parts[1].split("-")[0]
    return f"{company}-{model_name}"

def strip_thinking(text):
    import re
    try:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    except:
        return

responses = []
prompt = "What is the type of the following sentence: 'I want food'. Respond in only one word indicating the type"

for model in models:
    response, content = call_model(model, prompt)
    responses.append((extract_hfmodel_name(model), strip_thinking(content)))

pprint(responses)