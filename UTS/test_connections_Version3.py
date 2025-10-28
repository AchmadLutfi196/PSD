import os
from dotenv import load_dotenv
import psycopg2
import mysql.connector
from sqlalchemy import create_engine, text

def test_connections():
    """Test database connections"""
    load_dotenv()
    
    print("🔍 Testing Database Connections...")
    print("="*40)
    
    # Test PostgreSQL
    try:
        postgres_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        pg_engine = create_engine(postgres_url)
        
        with pg_engine.connect() as conn:
            result = conn.execute(text("SELECT version()")).fetchone()
            print(f"✅ PostgreSQL: Connected successfully")
            print(f"   Version: {result[0][:50]}...")
            
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
    
    # Test MySQL
    try:
        mysql_url = f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
        mysql_engine = create_engine(mysql_url)
        
        with mysql_engine.connect() as conn:
            result = conn.execute(text("SELECT VERSION()")).fetchone()
            print(f"✅ MySQL: Connected successfully")
            print(f"   Version: {result[0]}")
            
    except Exception as e:
        print(f"❌ MySQL: Connection failed - {e}")

if __name__ == "__main__":
    test_connections()