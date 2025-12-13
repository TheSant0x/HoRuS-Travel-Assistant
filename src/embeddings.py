import os
from neo4j import GraphDatabase
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
        
        # For now, we'll use a simple text-based similarity approach
        # Later this can be enhanced with actual embeddings
        self.dimension = 384 # Placeholder dimension
        
        Logger.log("Creating Vector Index...")
        self.create_vector_index()
        Logger.log("Populating Embeddings (this may take a while)...")
        self.populate_embeddings()
        Logger.log("Setup Complete.")

    def close(self):
        self.driver.close()

    def create_vector_index(self):
        """
        Creates a text index on the Hotel node for the 'search_text' property.
        """
        query = """
        CREATE INDEX hotel_search_text IF NOT EXISTS
        FOR (h:Hotel)
        ON (h.search_text)
        """
        with self.driver.session() as session:
            session.run(query)
            Logger.log("Text index 'hotel_search_text' created.")

    def populate_embeddings(self):
        """
        Fetches all hotels and creates simple text representations for search.
        Note: This is a simplified version without actual embeddings.
        """
        fetch_query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        RETURN h.hotel_id as id, h.name as name, h.star_rating as stars, 
               h.cleanliness_base as clean, h.comfort_base as comfort, 
               h.facilities_base as facilities, c.name as city, co.name as country
        """
        
        update_query = """
        MATCH (h:Hotel {hotel_id: $id})
        SET h.search_text = $search_text
        """
        
        with self.driver.session() as session:
            result = session.run(fetch_query)
            hotels = [record.data() for record in result]
            
            Logger.log(f"Creating search text for {len(hotels)} hotels...")
            
            for hotel in hotels:
                # Construct searchable text
                search_text = (
                    f"{hotel['name']} {hotel['city']} {hotel['country']} "
                    f"{hotel['stars']}star cleanliness{hotel['clean']} "
                    f"comfort{hotel['comfort']} facilities{hotel['facilities']}"
                ).lower()
                
                # Update Node with search text
                session.run(update_query, id=hotel['id'], search_text=search_text)
                
            Logger.log("Text indexing complete.")

    def search_similar_hotels(self, query_text: str, top_k: int = 3):
        """
        Simple text-based search using contains matching with keywords.
        """
        query_lower = query_text.lower()
        
        # Extract key search terms
        search_terms = []
        keywords = ["paris", "london", "tokyo", "hotel", "luxury", "budget", "star", "clean"]
        for keyword in keywords:
            if keyword in query_lower:
                search_terms.append(keyword)
        
        if not search_terms:
            # If no specific keywords, try to extract city names and other relevant words
            words = query_lower.replace(",", " ").replace(".", " ").split()
            search_terms = [word for word in words if len(word) > 3 and word not in ["find", "show", "hotels", "hotel", "give", "want"]]
        
        if not search_terms:
            return []
        
        # Build dynamic query for multiple search terms
        conditions = []
        params = {"k": top_k}
        
        for i, term in enumerate(search_terms):
            param_name = f"term_{i}"
            conditions.append(f"h.search_text CONTAINS ${param_name}")
            params[param_name] = term
        
        cypher = f"""
        MATCH (h:Hotel)
        WHERE {' OR '.join(conditions)}
        RETURN h.name as hotel, h.star_rating as stars, 
               h.average_reviews_score as rating, 
               0.8 as score
        ORDER BY h.star_rating DESC, h.average_reviews_score DESC
        LIMIT $k
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def format_results(self, results):
        if not results:
            return "No semantic matches found."
            
        formatted_lines = []
        for idx, res in enumerate(results):
            score = res.get('score', 0)
            formatted_lines.append(f"{idx+1}. {res.get('hotel')} (Score: {score:.4f})")
            
        return "\n".join(formatted_lines)
