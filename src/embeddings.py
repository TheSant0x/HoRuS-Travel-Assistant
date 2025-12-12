import os
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase

class EmbeddingManager:
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD")
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD not found in environment.")
            
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        # Initialize HuggingFace Embeddings (Local)
        # Using a standard lightweight model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.dimension = 384 # Dimension for all-MiniLM-L6-v2

    def close(self):
        self.driver.close()

    def create_vector_index(self):
        """
        Creates a vector index on the Hotel node for the 'embedding' property.
        """
        query = """
        CREATE VECTOR INDEX hotel_embeddings IF NOT EXISTS
        FOR (h:Hotel)
        ON (h.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: toInteger($dim),
            `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query, dim=self.dimension)
            print("Vector index 'hotel_embeddings' ensure created.")

    def populate_embeddings(self):
        """
        Fetches all hotels, constructs a text representation (Features Vector),
        generates embeddings, and writes them back to Neo4j.
        """
        fetch_query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        RETURN h.hotel_id as id, h.name as name, h.star_rating as stars, 
               h.cleanliness_base as clean, h.comfort_base as comfort, 
               h.facilities_base as facilities, c.name as city, co.name as country
        """
        
        update_query = """
        MATCH (h:Hotel {hotel_id: $id})
        CALL db.create.setNodeVectorProperty(h, 'embedding', $embedding)
        """
        
        with self.driver.session() as session:
            result = session.run(fetch_query)
            hotels = [record.data() for record in result]
            
            print(f"Generating embeddings for {len(hotels)} hotels...")
            
            for hotel in hotels:
                # Construct Feature Text
                # "Hotel: X. Located in Y, Z. Rated 5 stars. Cleanliness: 9.0..."
                text = (
                    f"Hotel: {hotel['name']}\n"
                    f"Location: {hotel['city']}, {hotel['country']}.\n"
                    f"Rating: {hotel['stars']} Stars.\n"
                    f"Features: Cleanliness {hotel['clean']}, Comfort {hotel['comfort']}, Facilities {hotel['facilities']}."
                )
                
                # Generate Embedding
                vector = self.embeddings.embed_query(text)
                
                # Update Node
                session.run(update_query, id=hotel['id'], embedding=vector)
                
            print("Embeddings population complete.")

    def search_similar_hotels(self, query_text: str, top_k: int = 3):
        """
        Embeds the user query and searches the vector index.
        """
        query_vector = self.embeddings.embed_query(query_text)
        
        cypher = """
        CALL db.index.vector.queryNodes('hotel_embeddings', $k, $p_vector)
        YIELD node, score
        RETURN node.name as hotel, score, node.star_rating as stars, node.average_reviews_score as rating
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, k=top_k, p_vector=query_vector)
            return [record.data() for record in result]

    def format_results(self, results):
        if not results:
            return "No semantic matches found."
            
        formatted_lines = []
        for idx, res in enumerate(results):
            score = res.get('score', 0)
            formatted_lines.append(f"{idx+1}. {res.get('hotel')} (Score: {score:.4f})")
            
        return "\n".join(formatted_lines)
