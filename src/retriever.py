import os
from neo4j import GraphDatabase

class GraphRetriever:
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD")
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD not found in environment.")
            
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        self.driver.close()

    def get_query_for_intent(self, intent_category: str, entities: dict) -> tuple[str, dict]:
        """
        Determines the appropriate Cypher query based on intent and present entities.
        Returns (query_string, parameters_dict).
        """
        # Unpack essential entities for decision making
        city = entities.get('city')
        hotel = entities.get('hotel_name')
        traveller_type = entities.get('traveller_type')
        min_rating = entities.get('min_rating')
        min_stars = entities.get('min_stars')
        attributes = entities.get('attributes', [])
        age_min = entities.get('age_min')
        target_country = entities.get('target_country')
        current_country = entities.get('current_country')
        
        # --- Intent: SEARCH ---
        if intent_category == "search":
            # Query 2: Specific Hotel
            if hotel:
                query = """
                MATCH (h:Hotel {name:$hotel_name})-[:LOCATED_IN]->(c:City)
                RETURN h.name as hotel, h.star_rating as stars, h.average_reviews_score as rating, c.name as city, 
                       h.cleanliness_base as cleanliness, h.comfort_base as comfort, h.facilities_base as facilities
                """
                return query, {"hotel_name": hotel}
            
            # Query 1: Hotels in City
            if city:
                query = """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city})
                RETURN h.name AS hotel, h.star_rating, h.average_reviews_score
                ORDER BY h.average_reviews_score DESC
                LIMIT 10
                """
                return query, {"city": city}
                
            # Query 10: Visa Check (Search for visa info)
            if target_country and current_country:
                query = """
                MATCH (c1:Country {name: $from_country})
                MATCH (c2:Country {name: $to_country})
                OPTIONAL MATCH (c1)-[v:NEEDS_VISA]->(c2)
                RETURN c1.name as from, c2.name as to, 
                       CASE WHEN v IS NULL THEN 'No Visa Required' ELSE v.visa_type END as visa_requirement
                """
                return query, {"from_country": current_country, "to_country": target_country}

        # --- Intent: RECOMMENDATION ---
        if intent_category == "recommendation":
            # Query 8: Age Demographics
            if age_min is not None:
                # Default max if not provided
                age_max = entities.get('age_max', age_min + 10) 
                query = """
                MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
                WHERE t.age >= $age_min AND t.age <= $age_max
                RETURN h.name AS hotel, avg(r.score_overall) AS rating
                ORDER BY rating DESC LIMIT 5
                """
                return query, {"age_min": age_min, "age_max": age_max}

            # Query 5: Traveller Type
            if traveller_type:
                query = """
                MATCH (t:Traveller {type:$traveller_type})-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
                RETURN h.name AS hotel, avg(r.score_overall) AS rating
                ORDER BY rating DESC LIMIT 10
                """
                return query, {"traveller_type": traveller_type}

            # Query 6: Facilities / Attributes (Clean, Comfort, etc)
            # Simple keyword mapping to base scores
            if attributes:
                # Default thresholds if not specified
                min_clean = 0.0
                min_comfort = 0.0
                min_fac = 0.0
                
                for attr in attributes:
                    a = attr.lower()
                    if "clean" in a: min_clean = 8.0
                    if "comfort" in a: min_comfort = 8.0
                    if "facilit" in a or "pool" in a or "wifi" in a: min_fac = 7.0 # Approximation for pool/wifi using facilities score

                query = """
                MATCH (h:Hotel)
                WHERE h.cleanliness_base >= $min_cleanliness
                  AND h.comfort_base >= $min_comfort
                  AND h.facilities_base >= $min_facilities
                RETURN h.name as hotel, h.star_rating, h.cleanliness_base, h.comfort_base, h.facilities_base
                ORDER BY h.star_rating DESC
                LIMIT 10
                """
                return query, {"min_cleanliness": min_clean, "min_comfort": min_comfort, "min_facilities": min_fac}

            # Query 9: Exceeds Expectations
            # Triggered if user asks for "exceed expectations" or "surprise me" or similar
            # Since we don't have a specific entity for this, we might need a flag or infer from query text.
            # For now, let's assume if no other specific criteria, we fallback to general top ranked or this.
            # But let's check input text or just use Query 7 (Top Rated) as default recommendation.
            
            # Query 4: Filter by Rating / Stars
            if min_rating or min_stars:
                query = """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
                WHERE h.average_reviews_score >= $minRating AND h.star_rating >= $minStars
                RETURN h.name as hotel, h.average_reviews_score, h.star_rating, c.name AS city
                ORDER BY h.average_reviews_score DESC
                LIMIT 10
                """
                params = {
                    "minRating": float(min_rating) if min_rating else 0.0,
                    "minStars": int(min_stars) if min_stars else 0
                }
                return query, params

            # Query 7: Top Rated (Default Recommendation)
            query = """
            MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)
            RETURN h.name AS hotel, avg(r.score_overall) AS rating
            ORDER BY rating DESC LIMIT 5
            """
            return query, {}

        # --- Intent: QUESTION (e.g. Reviews) ---
        if intent_category == "question":
            # Query 3: Reviews for Hotel
            if hotel:
                query = """
                MATCH (h:Hotel {name:$hotel_name})<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)
                RETURN r.text as review, r.date as date, r.score_overall as score, t.type AS traveller_type
                ORDER BY r.date DESC
                LIMIT 5
                """
                return query, {"hotel_name": hotel}

        return None, None

    def retrieve_baseline(self, intent_obj, entities_obj):
        """
        Executes a Cypher query based on the processed intent and entities.
        """
        intent_cat = intent_obj.category
        entities = entities_obj.dict()

        query, params = self.get_query_for_intent(intent_cat, entities)
        
        if not query:
            return []

        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def format_results(self, results):
        if not results:
            return "No direct matches found via Cypher."
        
        formatted_lines = []
        for idx, res in enumerate(results[:5]):
            # Format dictionary nicely or just stringify
            formatted_lines.append(f"{idx+1}. {res}")
        
        return "\n".join(formatted_lines)
