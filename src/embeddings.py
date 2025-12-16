import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from . import logger as Logger

class EmbeddingManager:
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD")
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD not found in environment.")
            
        Logger.log("Initializing Embedding Manager...")
        
        # Initialize database connection
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        # Initialize Sentence Transformer Models
        try:
            Logger.log("Loading Model 1: all-MiniLM-L6-v2 (384 dim)...")
            self.model_1 = SentenceTransformer('all-MiniLM-L6-v2')
            
            Logger.log("Loading Model 2: paraphrase-albert-small-v2 (768 dim)...")
            self.model_2 = SentenceTransformer('paraphrase-albert-small-v2')
        except Exception as e:
            Logger.log(f"Failed to load models: {e}", Logger.ERROR)
            raise e
        
        Logger.log("Creating Vector Indices...")
        self.create_vector_indices()
        
        Logger.log("Populating Embeddings (this may take a while)...")
        self.populate_embeddings()
        
        Logger.log("Setup Complete.")

    def close(self):
        self.driver.close()

    def create_vector_indices(self):
        """
        Creates Vector Indices for both embedding models.
        """
        queries = [
            """
            CREATE VECTOR INDEX hotel_embeddings IF NOT EXISTS
            FOR (h:Hotel)
            ON (h.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX hotel_embeddings_v2 IF NOT EXISTS
            FOR (h:Hotel)
            ON (h.embedding_v2)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}
            """
        ]
        
        with self.driver.session() as session:
            try:
                session.run(queries[0])
                Logger.log("Vector index 'hotel_embeddings' (384d) verified.")
            except Exception as e:
                Logger.log(f"Error creating index 1: {e}", Logger.ERROR)

            try:
                session.run(queries[1])
                Logger.log("Vector index 'hotel_embeddings_v2' (768d) verified.")
            except Exception as e:
                Logger.log(f"Error creating index 2: {e}", Logger.ERROR)

    def populate_embeddings(self):
        """
        Fetches all hotels, creates rich text, generates both embeddings, and updates the graph.
        """
        fetch_query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        RETURN h.hotel_id as id, h.name as name, h.star_rating as stars, 
               h.cleanliness_base as clean, h.comfort_base as comfort, 
               h.facilities_base as facilities, c.name as city, co.name as country
        """
        
        update_query = """
        MATCH (h:Hotel {hotel_id: $id})
        SET h.embedding = $embedding,
            h.embedding_v2 = $embedding_v2,
            h.search_text = $search_text
        """
        
        with self.driver.session() as session:
            result = session.run(fetch_query)
            hotels = [record.data() for record in result]
            
            Logger.log(f"Generating dual embeddings for {len(hotels)} hotels...")
            
            batch_count = 0
            for hotel in hotels:
                search_text = (
                    f"Hotel {hotel['name']} in {hotel['city']}, {hotel['country']}. "
                    f"{hotel['stars']} star rating. "
                    f"Cleanliness score: {hotel['clean']}. "
                    f"Comfort score: {hotel['comfort']}. "
                    f"Facilities score: {hotel['facilities']}."
                )
                
                # Generate Embeddings
                emb_1 = self.model_1.encode(search_text).tolist()
                emb_2 = self.model_2.encode(search_text).tolist()
                
                # Update Node
                session.run(update_query, id=hotel['id'], 
                            embedding=emb_1, 
                            embedding_v2=emb_2, 
                            search_text=search_text)
                            
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Processed {batch_count}/{len(hotels)} hotels...", end='\r')
                    
            print(f"Processed {batch_count}/{len(hotels)} hotels. Done.")
            Logger.log("Dual embeddings population complete.")

    def search_similar_hotels(self, query_text: str, top_k: int = 3, model_version: int = 1):
        """
        Semantic search using vector similarity with specified model version.
        """
        if not query_text:
            return []
            
        index_name = 'hotel_embeddings' if model_version == 1 else 'hotel_embeddings_v2'
        model = self.model_1 if model_version == 1 else self.model_2
        
        # 1. Generate embedding for query
        query_embedding = model.encode(query_text).tolist()
        
        # 2. Query the Vector Index
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
        YIELD node, score
        RETURN node.name as hotel,
               node.star_rating as stars,
               node.average_reviews_score as rating,
               score
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, k=top_k, embedding=query_embedding)
            return [record.data() for record in result]

    def format_results(self, results):
        if not results:
            return "No semantic matches found."
            
        formatted_lines = []
        for idx, res in enumerate(results):
            score = res.get('score', 0)
            formatted_lines.append(f"{idx+1}. {res.get('hotel')} (Similarity: {score:.4f})")
            
        return "\n".join(formatted_lines)
