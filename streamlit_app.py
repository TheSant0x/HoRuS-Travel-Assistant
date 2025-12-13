import streamlit as st
import os
import sys
import json
from typing import Dict, Any, List
import time
import traceback

# Add the project root to the path so we can import our modules
sys.path.append('.')

from src.processor import Preprocessor
from src.retriever import GraphRetriever
from src.embeddings import EmbeddingManager
import src.logger as Logger
import src.inference as Inference
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Graph-RAG Travel Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #ff7f0e;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .cypher-query {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fed7d7;
        color: #c53030;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #e53e3e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitTravelAssistant:
    def __init__(self):
        self.processor = None
        self.retriever = None
        self.embedder = None
        self.initialized = False
        
    def initialize_components(self):
        """Initialize all components if not already done"""
        if not self.initialized:
            try:
                with st.spinner("Initializing Graph-RAG components..."):
                    Logger.verbosity = 1  # Set appropriate verbosity
                    self.processor = Preprocessor()
                    self.retriever = GraphRetriever()
                    self.embedder = EmbeddingManager()
                    self.initialized = True
                st.success("Components initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize components: {str(e)}")
                st.error("Please check your Neo4j connection and environment variables.")
                return False
        return True
    
    def check_environment(self):
        """Check if required environment variables are set"""
        required_vars = ["HF_TOKEN", "NEO4J_PASSWORD", "NEO4J_URI"]
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            st.error(f"Missing environment variables: {', '.join(missing_vars)}")
            st.info("Please set these variables in your .env file")
            return False
        return True
    
    def process_query(self, query: str, model_name: str, retrieval_method: str) -> Dict[str, Any]:
        """Process a single query and return structured results"""
        results = {
            "intent": None,
            "entities": {},
            "baseline_results": [],
            "embedding_results": [],
            "cypher_queries": [],
            "final_answer": "",
            "error": None,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze request
            intent, entities = self.processor.process(query)
            results["intent"] = intent.category
            results["entities"] = entities.model_dump()
            
            # Step 2: Retrieve from Knowledge Graph
            baseline_results = []
            embedding_results = []
            
            if retrieval_method in ["baseline", "both"]:
                baseline_results = self.retriever.retrieve_baseline(intent, entities)
                # Get the executed queries for display
                if hasattr(self.retriever, 'last_queries'):
                    results["cypher_queries"] = self.retriever.last_queries
            
            if retrieval_method in ["embeddings", "both"]:
                if intent.category in ["search", "recommendation"]:
                    embedding_results = self.embedder.search_similar_hotels(query)
            
            results["baseline_results"] = baseline_results
            results["embedding_results"] = embedding_results
            
            # Step 3: Generate LLM response
            if retrieval_method == "both":
                context = baseline_results + embedding_results
            elif retrieval_method == "embeddings":
                context = embedding_results
            else:
                context = baseline_results
            
            if context:
                formatted_query = Inference.format_prompt(query, context)
                client = Inference.setup_inference()
                response = Inference.call_model(client, model_name, formatted_query)
                results["final_answer"] = response
            else:
                results["final_answer"] = "No relevant information found in the knowledge graph."
            
        except Exception as e:
            results["error"] = str(e)
            results["final_answer"] = f"Error processing query: {str(e)}"
        
        results["processing_time"] = time.time() - start_time
        return results
    
    def display_results(self, results: Dict[str, Any]):
        """Display the results in a structured format"""
        
        # Processing Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Intent", results["intent"] or "Unknown")
        with col2:
            st.metric("Processing Time", f"{results['processing_time']:.2f}s")
        with col3:
            entities_count = len([v for v in results["entities"].values() if v])
            st.metric("Entities Found", entities_count)
        
        # Entities
        if any(results["entities"].values()):
            st.markdown('<div class="section-header">Extracted Entities</div>', unsafe_allow_html=True)
            entities_text = ", ".join([f"{k}: {v}" for k, v in results["entities"].items() if v])
            st.markdown(f'<div class="result-box">{entities_text}</div>', unsafe_allow_html=True)
        
        # Cypher Queries
        if results["cypher_queries"]:
            st.markdown('<div class="section-header">Executed Cypher Queries</div>', unsafe_allow_html=True)
            for i, query in enumerate(results["cypher_queries"], 1):
                st.markdown(f'<div class="cypher-query">Query {i}:<br>{query}</div>', unsafe_allow_html=True)
        
        # Results Tabs
        tab1, tab2, tab3 = st.tabs(["KG Retrieved Context", "Final LLM Answer", "Detailed Results"])
        
        with tab1:
            st.markdown('<div class="section-header">Knowledge Graph Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Baseline Results (Cypher)")
                if results["baseline_results"]:
                    formatted_baseline = self.retriever.format_results(results["baseline_results"])
                    st.markdown(f'<div class="result-box">{formatted_baseline}</div>', unsafe_allow_html=True)
                    
                    # Show raw data
                    with st.expander("View Raw Baseline Data"):
                        st.json(results["baseline_results"])
                else:
                    st.info("No baseline results found")
            
            with col2:
                st.subheader("Embedding Results (Semantic Search)")
                if results["embedding_results"]:
                    formatted_embeddings = self.embedder.format_results(results["embedding_results"])
                    st.markdown(f'<div class="result-box">{formatted_embeddings}</div>', unsafe_allow_html=True)
                    
                    # Show raw data
                    with st.expander("View Raw Embedding Data"):
                        st.json(results["embedding_results"])
                else:
                    st.info("No embedding results found")
        
        with tab2:
            st.markdown('<div class="section-header">Final Answer</div>', unsafe_allow_html=True)
            if results["error"]:
                st.markdown(f'<div class="error-box">Error: {results["error"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box">{results["final_answer"]}</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Result Statistics")
                st.write(f"**Baseline Results Count:** {len(results['baseline_results'])}")
                st.write(f"**Embedding Results Count:** {len(results['embedding_results'])}")
                st.write(f"**Queries Executed:** {len(results['cypher_queries'])}")
                st.write(f"**Total Processing Time:** {results['processing_time']:.2f} seconds")
            
            with col2:
                st.subheader("Technical Details")
                st.json({
                    "intent": results["intent"],
                    "entities": results["entities"],
                    "has_baseline_results": len(results["baseline_results"]) > 0,
                    "has_embedding_results": len(results["embedding_results"]) > 0,
                    "error_occurred": results["error"] is not None
                })

def main():
    # Header
    st.markdown('<div class="main-header">Graph-RAG Travel Assistant</div>', unsafe_allow_html=True)
    st.markdown("*Powered by Neo4j Knowledge Graph and Large Language Models*")
    
    # Initialize the assistant
    assistant = StreamlitTravelAssistant()
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # Environment Check
    if not assistant.check_environment():
        st.stop()
    
    # Model Selection
    st.sidebar.subheader("Model Selection")
    model_options = [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small", 
        "gpt2",
        "distilgpt2"
    ]
    selected_model = st.sidebar.selectbox(
        "Choose LLM Model:", 
        model_options,
        help="Select the language model for generating responses"
    )
    
    # Retrieval Method Selection
    st.sidebar.subheader("Retrieval Method")
    retrieval_method = st.sidebar.selectbox(
        "Choose Retrieval Strategy:",
        ["baseline", "embeddings", "both"],
        index=2,
        help="Select how to retrieve information from the knowledge graph"
    )
    
    # Method descriptions
    method_descriptions = {
        "baseline": "Uses structured Cypher queries for exact matches",
        "embeddings": "Uses semantic similarity search with embeddings", 
        "both": "Combines both baseline and embedding approaches"
    }
    st.sidebar.info(f"**{retrieval_method.title()}:** {method_descriptions[retrieval_method]}")
    
    # Sample Queries
    st.sidebar.subheader("Sample Queries")
    sample_queries = [
        "Find hotels in Paris",
        "Show me luxury hotels in France", 
        "What are the best hotels with good cleanliness ratings?",
        "Recommend hotels in Tokyo",
        "Find 5-star hotels with high comfort ratings"
    ]
    
    for query in sample_queries:
        if st.sidebar.button(query, key=f"sample_{hash(query)}"):
            st.session_state.query_input = query
    
    # Initialize components
    if not assistant.initialize_components():
        st.stop()
    
    # Main Query Interface
    st.markdown("---")
    st.subheader("Ask Your Travel Question")
    
    # Query input
    query = st.text_input(
        "Enter your travel-related question:",
        value=st.session_state.get('query_input', ''),
        placeholder="e.g., Find hotels in Paris with good ratings",
        help="Ask about hotels, locations, ratings, or get recommendations"
    )
    
    # Process button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        process_button = st.button("Process Query", type="primary", use_container_width=True)
    
    # Process the query
    if process_button and query:
        st.markdown("---")
        
        with st.spinner("Processing your query..."):
            results = assistant.process_query(query, selected_model, retrieval_method)
        
        # Display results
        assistant.display_results(results)
        
        # Store results in session state for comparison
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        st.session_state.query_history.append({
            'query': query,
            'model': selected_model,
            'method': retrieval_method,
            'results': results,
            'timestamp': time.time()
        })
        
        # Limit history to last 5 queries
        if len(st.session_state.query_history) > 5:
            st.session_state.query_history = st.session_state.query_history[-5:]
    
    elif process_button and not query:
        st.warning("Please enter a query first!")
    
    # Query History and Comparison
    if 'query_history' in st.session_state and st.session_state.query_history:
        st.markdown("---")
        st.subheader("Query History & Model Comparison")
        
        # Display history in expandable sections
        for i, history_item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query {len(st.session_state.query_history)-i}: {history_item['query'][:50]}... | Model: {history_item['model']} | Method: {history_item['method']}"):
                assistant.display_results(history_item['results'])
    
    # Footer with system information
    st.markdown("---")
    st.markdown("**System Status:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("Neo4j Connected")
    with col2:
        st.success("Models Available")
    with col3:
        st.success("Embeddings Ready")

if __name__ == "__main__":
    main()
