import os
import sys
import argparse
from dotenv import load_dotenv
from src.processor import Preprocessor
from src.retriever import GraphRetriever
from src.embeddings import EmbeddingManager

def setup_embeddings():
    print("Initializing Embedding Manager...")
    em = EmbeddingManager()
    print("Creating Vector Index...")
    em.create_vector_index()
    print("Populating Embeddings (this may take a while)...")
    em.populate_embeddings()
    em.close()
    print("Setup Complete.")

def main():
    load_dotenv()
    
    # CLI Argument to run setup
    parser = argparse.ArgumentParser(description="Graph RAG Travel Assistant")
    parser.add_argument("--setup", action="store_true", help="Initialize Vector Embeddings in Neo4j")
    args = parser.parse_args()
    
    if args.setup:
        setup_embeddings()
        return

    # Check keys
    if not os.environ.get("HF_TOKEN"):
        print("[!] Error: HF_TOKEN is missing in .env")
        return
    if not os.environ.get("NEO4J_PASSWORD"):
        print("[!] Error: NEO4J_PASSWORD is missing in .env")
        return

    try:
        print("Initializing Components...")
        processor = Preprocessor()
        retriever = GraphRetriever()
        embedder = EmbeddingManager()
        
        print("\n=== Graph-RAG Travel Assistant (M3) ===")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("\nUser: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue

            print("...> Analyzing Request...")
            intent, entities = processor.process(user_input)
            
            print(f"    [Intent]: {intent.category}")
            print(f"    [Entities]: {', '.join([f'{k}={v}' for k,v in entities.dict().items() if v])}")
            
            print("...> Retrieving from Knowledge Graph...")
            
            # 1. Baseline Retrieval
            baseline_results = retriever.retrieve_baseline(intent, entities)
            
            # 2. Embedding Retrieval (Vector Search)
            embedding_results = []
            if intent.category in ["search", "recommendation"]:
                embedding_results = embedder.search_similar_hotels(user_input)

            # Display Results
            print("\n--- Baseline Results (Cypher) ---")
            print(retriever.format_results(baseline_results))

            print("\n--- Semantic Search Results (Embeddings) ---")
            print(embedder.format_results(embedding_results))
            
            print("\n" + "="*50)

        retriever.close()
        embedder.close()

    except Exception as e:
        print(f"Application Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
