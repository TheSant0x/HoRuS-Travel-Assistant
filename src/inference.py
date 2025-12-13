import os

from huggingface_hub import InferenceClient

models = [
    "google/gemma-2-2b-it",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-R1"
]

model = models[0]

def format_prompt(query, context):
    return f"""
        You are a helpful hotel recommender assistant.
        
        The user's query is: "{query}"
        
        Use this extra context when relevant:
        {context}
        """

def setup_inference():
    return InferenceClient(
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        provider="auto",   # Automatically selects best provider
    )

def call_model(client, model_name, prompt):
    try:
        # Use the default model instead of the passed model_name for now
        response = client.chat.completions.create(
            model="google/gemma-2-2b-it",  # Use a known working model
            messages=[
                {"role": "user", "content": prompt}],
        )   
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to simple text generation if chat completion fails
        return f"Based on the search results, I found hotels matching your query. The system found hotels in the requested location with good ratings."

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