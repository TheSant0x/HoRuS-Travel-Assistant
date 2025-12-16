import os
import json
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models import Intent, Entities

# Initialize LLM
# Using HuggingFaceEndpoint for direct inference
repo_id = os.environ.get("HF_REPO_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
hf_token = os.environ.get("HF_TOKEN")

try:
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )
    llm = ChatHuggingFace(llm=endpoint)
except Exception as e:
    # Fallback or error handling if init fails (e.g. missing token)
    print(f"Failed to initialize HuggingFaceEndpoint: {e}")
    llm = None

class Preprocessor:
    def __init__(self):
        if not llm:
            raise ValueError("LLM is not initialized. Check your HF_TOKEN.")
            
        self.intent_parser = JsonOutputParser(pydantic_object=Intent)
        self.intent_chain = self._build_intent_chain()
        
        self.entity_parser = JsonOutputParser(pydantic_object=Entities)
        self.entity_chain = self._build_entity_chain()

    def _build_intent_chain(self):
        system_prompt = """You are an expert intent classifier for a Travel Assistant.
Analyze the user's query and classify it into one of the following categories:
- question: The user is asking for a specific fact (e.g., "Does Hotel X have a pool?", "Where is Paris?").
- recommendation: The user is asking for suggestions (e.g., "Suggest a romantic hotel", "Where should I stay?").
- search: The user is searching for a specific entity entry (e.g., "Show me the Hilton", "Find user 123", "List hotels").
- greeting: The user is greeting the assistant (e.g., "hello", "hi", "hola", "good morning").

You must output a valid JSON object matching the following structure:
{format_instructions}

Category must be exactly one of: "question", "recommendation", "search", "greeting".
IMPORTANT: Output ONLY the JSON object. Do not output any explanation or preamble.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        # Inject format instructions
        prompt = prompt.partial(format_instructions=self.intent_parser.get_format_instructions())
        
        return prompt | llm | self.intent_parser

    def _build_entity_chain(self):
        system_prompt = """You are an expert Named Entity Recognizer (NER) for a Hotel Travel Assistant.
Extract the following entities from the user's query into a JSON object:

- city: Target city name (e.g. "Paris").
- country: Target country name (e.g. "France").
- hotel_name: Specific hotel name (e.g. "Hilton").
- traveller_type: ONE of ["Solo", "Couple", "Family", "Business"] if mentioned or implied.
- min_rating: Minimum review score number (e.g. "rated above 8").
- min_stars: Minimum star rating (e.g. "5 star hotel").
- min_cleanliness: Minimum cleanliness score.
- min_comfort: Minimum comfort score.
- min_facilities: Minimum facilities score.
- age_min / age_max: Age range for demographics (e.g. "for people aged 18-24" -> min=18, max=24).
- target_country: Country to visit (for visa queries).
- current_country: User's origin country (for visa queries).
- attributes: general attributes like "pool", "wifi".

Only extract what is explicitly mentioned.
You must output a valid JSON object matching the following structure:
{format_instructions}

IMPORTANT: Output ONLY the JSON object. Do not output any explanation or preamble.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        prompt = prompt.partial(format_instructions=self.entity_parser.get_format_instructions())
        
        return prompt | llm | self.entity_parser

    def process(self, query: str):
        print(f"Processing query: '{query}'")
        # Invoke chains
        intent_data = self.intent_chain.invoke({"query": query})
        
        # Convert dict back to Pydantic model for consistency if needed, 
        # or rely on the parser output which is usually a dict.
        # The JsonOutputParser returns a dict, but we want the Pydantic object for the rest of the app?
        # The original code returned 'Intent' object. Let's validate.
        if isinstance(intent_data, dict):
            intent = Intent(**intent_data)
        else:
            intent = intent_data

        entities_data = self.entity_chain.invoke({"query": query})
        if isinstance(entities_data, dict):
            entities = Entities(**entities_data)
        else:
            entities = entities_data
            
        return intent, entities

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not found in environment. Please set it in .env")
    else:
        try:
            processor = Preprocessor()
            queries = [
                "Find me a cheap hotel in Paris with a pool.",
                "Does the Hilton in London have good reviews?",
                "Suggest a place for a family trip to Italy."
            ]
            
            for q in queries:
                try:
                    i, e = processor.process(q)
                    print(f"\nQuery: {q}")
                    print(f"Intent: {i.category} ({i.reasoning})")
                    print(f"Entities: {e.dict()}")
                except Exception as e:
                    print(f"Failed query '{q}': {e}")
        except Exception as e:
            print(f"Setup failed: {e}")
