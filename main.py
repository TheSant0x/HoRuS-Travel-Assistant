import os

from dotenv import load_dotenv

from src.processor import Preprocessor
from src.retriever import GraphRetriever
from src.embeddings import EmbeddingManager
import src.logger as Logger
import src.inference as Inference

load_dotenv()

def get_response(model_name, verbosity, query, add_embeddings):

    Logger.verbosity = verbosity
    
    if add_embeddings:
        EmbeddingManager() # only needs to be instantiated
        return

    # Check keys
    if not os.environ.get("HF_TOKEN"):
        Logger.log("[!] Error: HF_TOKEN is missing in .env", Logger.ERROR)
        return
    if not os.environ.get("NEO4J_PASSWORD"):
        Logger.log("[!] Error: NEO4J_PASSWORD is missing in .env", Logger.ERROR)
        return

    try:
        Logger.log("Initializing Components...")
        processor = Preprocessor()
        retriever = GraphRetriever()
        embedder = EmbeddingManager()
        
        if query:
            # Single query mode
            Logger.log(f"Processing query: {query}")
            Logger.log("...> Analyzing Request...")
            intent, entities = processor.process(query)
            
            Logger.log(f"    [Intent]: {intent.category}")
            Logger.log(f"    [Entities]: {', '.join([f'{k}={v}' for k,v in entities.model_dump().items() if v])}")
            
            Logger.log("...> Retrieving from Knowledge Graph...")
            
            # 1. Baseline Retrieval
            baseline_results = retriever.retrieve_baseline(intent, entities)
            
            # 2. Embedding Retrieval (Vector Search)
            embedding_results = []
            if intent.category in ["search", "recommendation"]:
                embedding_results = embedder.search_similar_hotels(query)

            # Display Results
            Logger.log("\n--- Baseline Results (Cypher) ---")
            Logger.log(retriever.format_results(baseline_results))

            Logger.log("\n--- Semantic Search Results (Embeddings) ---")
            Logger.log(embedder.format_results(embedding_results))
            
            context = baseline_results + embedding_results if add_embeddings else baseline_results
            
            formatted_query = Inference.format_prompt(query, context)
            client = Inference.setup_inference()
            
            response = Inference.call_model(client, model_name, formatted_query)
            
            retriever.close()
            embedder.close()
            return response
            
        else:
            # Interactive mode
            Logger.log("\n=== Graph-RAG Travel Assistant (M3) ===")
            Logger.log("Type 'exit' to quit.\n")
            
            while True:
                user_input = input("\nUser: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                if not user_input.strip():
                    continue

                Logger.log("...> Analyzing Request...")
                intent, entities = processor.process(user_input)
                
                Logger.log(f"    [Intent]: {intent.category}")
                Logger.log(f"    [Entities]: {', '.join([f'{k}={v}' for k,v in entities.model_dump().items() if v])}")
                
                Logger.log("...> Retrieving from Knowledge Graph...")
                
                # 1. Baseline Retrieval
                baseline_results = retriever.retrieve_baseline(intent, entities)
                
                # 2. Embedding Retrieval (Vector Search)
                embedding_results = []
                if intent.category in ["search", "recommendation"]:
                    embedding_results = embedder.search_similar_hotels(user_input)

                # Display Results
                Logger.log("\n--- Baseline Results (Cypher) ---")
                Logger.log(retriever.format_results(baseline_results))

                Logger.log("\n--- Semantic Search Results (Embeddings) ---")
                Logger.log(embedder.format_results(embedding_results))
                
                Logger.log("\n" + "="*50)

            retriever.close()
            embedder.close()

    except Exception as e:
        Logger.log(f"Application Error: {e}", Logger.ERROR)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph-RAG Travel Assistant")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", 
                       help="Model name for inference (default: microsoft/DialoGPT-medium)")
    parser.add_argument("--verbosity", type=int, default=1, 
                       help="Verbosity level (0=minimal, 1=normal, 2=detailed)")
    parser.add_argument("--query", type=str, 
                       help="Single query to process (if not provided, starts interactive mode)")
    parser.add_argument("--add-embeddings", action="store_true", 
                       help="Only add embeddings to the database and exit")
    
    args = parser.parse_args()
    
    if args.add_embeddings:
        print("Adding embeddings to the database...")
        get_response(args.model, args.verbosity, "", True)
        print("Embeddings added successfully!")
    elif args.query:
        print(f"Processing query: {args.query}")
        response = get_response(args.model, args.verbosity, args.query, False)
        if response:
            print(f"Response: {response}")
    else:
        # Interactive mode
        print("Starting interactive mode...")
        get_response(args.model, args.verbosity, "", False)