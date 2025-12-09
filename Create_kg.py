from neo4j import GraphDatabase
import csv
import os
from dotenv import load_dotenv

def read_config(config_file='config.txt'):
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value
    return config

def clear_database(tx):
    # Delete in batches to avoid timeout on large datasets
    query = """
    MATCH (n)
    WITH n LIMIT 10000
    DETACH DELETE n
    RETURN count(n) as deleted_count
    """
    result = tx.run(query)
    record = result.single()
    return record['deleted_count'] if record else 0

def clear_database_loop(session):
    total_deleted = 0
    while True:
        deleted = session.execute_write(clear_database)
        total_deleted += deleted
        print(f"Deleted {deleted} nodes...")
        if deleted == 0:
            break
    print(f"Total deleted: {total_deleted}")

def create_constraints(tx):
    # Create uniqueness constraints (or indexes) for faster lookups and data integrity
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE")

def load_hotels(tx, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
        
        # Batch insert Hotels
        query = """
        UNWIND $batch as row
        MERGE (c:Country {name: row.country})
        MERGE (ci:City {name: row.city})
        MERGE (ci)-[:LOCATED_IN]->(c)
        MERGE (h:Hotel {hotel_id: toInteger(row.hotel_id)})
        SET h.name = row.hotel_name,
            h.star_rating = toFloat(row.star_rating),
            h.cleanliness_base = toFloat(row.cleanliness_base),
            h.comfort_base = toFloat(row.comfort_base),
            h.facilities_base = toFloat(row.facilities_base)
        MERGE (h)-[:LOCATED_IN]->(ci)
        """
        tx.run(query, batch=batch)

def load_users(tx, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        count = 0
        for row in reader:
            batch.append(row)
            if len(batch) >= 500:
                _run_user_batch(tx, batch)
                count += len(batch)
                print(f"Loaded {count} users...", end='\r')
                batch = []
        if batch:
            _run_user_batch(tx, batch)
            count += len(batch)
    print(f"Loaded {count} users. Done.")

def _run_user_batch(tx, batch):
    query = """
    UNWIND $batch as row
    MERGE (c:Country {name: row.country})
    MERGE (t:Traveller {user_id: toInteger(row.user_id)})
    SET t.age = row.age_group,
        t.type = row.traveller_type,
        t.gender = row.user_gender
    MERGE (t)-[:FROM_COUNTRY]->(c)
    """
    tx.run(query, batch=batch)

def load_reviews(driver, file_path):
    # Note: We pass 'driver' instead of 'tx' because we manage transactions manually here
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        count = 0
        for row in reader:
            batch.append(row)
            if len(batch) >= 100:  # Reduced batch size to 100
                with driver.session() as session:
                    session.execute_write(_run_review_batch, batch)
                count += len(batch)
                print(f"Loaded {count} reviews...", end='\r')
                batch = []
        if batch:
            with driver.session() as session:
                session.execute_write(_run_review_batch, batch)
            count += len(batch)
    print(f"Loaded {count} reviews. Done.")

def _run_review_batch(tx, batch):
    query = """
    UNWIND $batch as row
    MATCH (t:Traveller {user_id: toInteger(row.user_id)})
    MATCH (h:Hotel {hotel_id: toInteger(row.hotel_id)})
    MERGE (r:Review {review_id: toInteger(row.review_id)})
    SET r.text = row.review_text,
        r.date = row.review_date,
        r.score_overall = toFloat(row.score_overall),
        r.score_cleanliness = toFloat(row.score_cleanliness),
        r.score_comfort = toFloat(row.score_comfort),
        r.score_facilities = toFloat(row.score_facilities),
        r.score_location = toFloat(row.score_location),
        r.score_staff = toFloat(row.score_staff),
        r.score_value_for_money = toFloat(row.score_value_for_money)
    MERGE (t)-[:WROTE]->(r)
    MERGE (r)-[:REVIEWED]->(h)
    MERGE (t)-[:STAYED_AT]->(h)
    """
    tx.run(query, batch=batch)

def load_visa(tx, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
        
        query = """
        UNWIND $batch as row
        MERGE (c1:Country {name: row.from})
        MERGE (c2:Country {name: row.to})
        MERGE (c1)-[v:NEEDS_VISA]->(c2)
        SET v.visa_type = row.visa_type
        """
        tx.run(query, batch=batch)

def compute_hotel_scores(tx):
    query = """
    MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)
    WITH h, avg(r.score_overall) as avg_score
    SET h.average_reviews_score = avg_score
    """
    tx.run(query)

def main():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, username, password]):
        print("Error: Missing Neo4j credentials in .env")
        return

    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        print("Clearing database...")
        clear_database_loop(session)
        
        print("Creating constraints...")
        session.execute_write(create_constraints)
        
        print("Loading Hotels...")
        session.execute_write(load_hotels, 'hotels.csv')
        
        print("Loading Users...")
        session.execute_write(load_users, 'users.csv')
        
        print("Loading Reviews...")
        # load_reviews manages its own sessions/transactions now
        load_reviews(driver, 'reviews.csv')
        
        print("Loading Visa data...")
        session.execute_write(load_visa, 'visa.csv')
        
        print("Computing Hotel Average Scores...")
        session.execute_write(compute_hotel_scores)
        
        print("Knowledge Graph created successfully!")

    driver.close()

if __name__ == "__main__":
    main()