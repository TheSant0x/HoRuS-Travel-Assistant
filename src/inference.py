import os

from huggingface_hub import InferenceClient

models = [
    "google/gemma-2-2b-it",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-R1"
]

model = models[0]

def format_prompt(query, context):
    context_str = ""
    if context:
        # Check if context is a list of dicts and format it
        if isinstance(context, list) and context and isinstance(context[0], dict):
            lines = []
            for item in context:
                # Create a readable string for each hotel/item
                # Filter out internal keys like 'score' if needed, or just format nicely
                details = ", ".join([f"{k}: {v}" for k, v in item.items() if k != 'score'])
                lines.append(f"- {details}")
            context_str = "\n".join(lines)
        else:
            context_str = str(context)

    template = f"""
    You are a helpful hotel recommender assistant. 
    Answer the user's query based on the provided context.
    Do not output raw JSON. Provide a natural language response summarizing the recommendations.
    
    IMPORTANT: Do NOT ask any follow-up questions. Do NOT ask for more preferences. Just provide the recommendations based on what you know.
    
    User Query: "{query}"
    
    Context (Available Hotels):
    {context_str}
    
    Response:
    """
    return template

def setup_inference():
    return InferenceClient(
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        provider="auto",   # Automatically selects best provider
    )

def call_model(client, model_name, prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use the passed model_name
            print(f"DEBUG: Using model: {model_name}")
            response = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "user", "content": prompt}],
                max_tokens=500
            )   
            response_text = response.choices[0].message.content
            return strip_thinking(response_text)
        except Exception as e:
            if attempt == max_retries - 1:
                # If it's the last attempt, raise the error so the app can handle it
                raise e
            # Wait a bit before retrying
            import time
            time.sleep(1)

def extract_hfmodel_name(model):
    parts = model.split("/")
    company = parts[0]
    model_name = parts[1].split("-")[0]
    return f"{company}-{model_name}"

def strip_thinking(text):
    import re
    try:
        # Remove <think> tags and content
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    except:
        return text