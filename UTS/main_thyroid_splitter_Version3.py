import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import mysql.connector
import psycopg2
from dotenv import load_dotenv
import os
import logging
from io import StringIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThyroidDataSplitter:
    def __init__(self):
        load_dotenv()
        self.setup_connections()
        
    def setup_connections(self):
        """Setup database connections dengan error handling"""
        try:
            # PostgreSQL connection
            postgres_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
            self.postgres_engine = create_engine(postgres_url)
            
            # Test PostgreSQL connection
            with self.postgres_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ PostgreSQL connection successful")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            raise
            
        try:
            # MySQL connection
            mysql_url = f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
            self.mysql_engine = create_engine(mysql_url)
            
            # Test MySQL connection
            with self.mysql_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ MySQL connection successful")
            
        except Exception as e:
            logger.error(f"❌ MySQL connection failed: {e}")
            raise
    
    def load_sample_data(self):
        """Load sample thyroid data"""
        # Data sample yang lebih lengkap
        csv_content = """Age,Sex,On_thyroxine,Query_on_thyroxine,On_antithyroid_medication,Sick,Pregnant,Thyroid_surgery,I131_treatment,Query_hypothyroid,Query_hyperthyroid,Lithium,Goitre,Tumor,Hypopituitary,Psych,TSH,T3,TT4,T4U,FTI,Class
0.73,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0.0006,0.015,0.12,0.082,0.146,3
0.24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00025,0.03,0.143,0.133,0.108,3
0.47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0019,0.024,0.102,0.131,0.078,3
0.64,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0009,0.017,0.077,0.09,0.085,3
0.23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00025,0.026,0.139,0.09,0.153,3
0.45,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0012,0.019,0.098,0.075,0.131,2
0.56,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0.0018,0.022,0.156,0.145,0.108,1
0.78,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0.0008,0.016,0.089,0.098,0.091,2
0.34,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0.0015,0.028,0.134,0.112,0.119,1
0.67,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0.0011,0.021,0.167,0.156,0.107,3"""
        
        df = pd.read_csv(StringIO(csv_content))
        logger.info(f"📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def create_tables(self):
        """Create tables in both databases"""
        # PostgreSQL table for categorical features
        postgres_schema = """
        DROP TABLE IF EXISTS thyroid_categorical;
        CREATE TABLE thyroid_categorical (
            id SERIAL PRIMARY KEY,
            sex INTEGER,
            on_thyroxine INTEGER,
            query_on_thyroxine INTEGER,
            on_antithyroid_medication INTEGER,
            sick INTEGER,
            pregnant INTEGER,
            thyroid_surgery INTEGER,
            i131_treatment INTEGER,
            query_hypothyroid INTEGER,
            query_hyperthyroid INTEGER,
            lithium INTEGER,
            goitre INTEGER,
            tumor INTEGER,
            hypopituitary INTEGER,
            psych INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # MySQL table for numerical features and labels
        mysql_schema = """
        DROP TABLE IF EXISTS thyroid_numerical;
        CREATE TABLE thyroid_numerical (
            id INT AUTO_INCREMENT PRIMARY KEY,
            age DECIMAL(10,6),
            tsh DECIMAL(15,8),
            t3 DECIMAL(10,6),
            tt4 DECIMAL(10,6),
            t4u DECIMAL(10,6),
            fti DECIMAL(10,6),
            class_label INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            with self.postgres_engine.connect() as conn:
                conn.execute(text(postgres_schema))
                conn.commit()
                logger.info("✅ PostgreSQL table created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating PostgreSQL table: {e}")
            raise
        
        try:
            with self.mysql_engine.connect() as conn:
                conn.execute(text(mysql_schema))
                conn.commit()
                logger.info("✅ MySQL table created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating MySQL table: {e}")
            raise
    
    def split_and_store_data(self, df):
        """Split data and store in respective databases"""
        # Define feature categories
        categorical_features = [
            'Sex', 'On_thyroxine', 'Query_on_thyroxine', 'On_antithyroid_medication',
            'Sick', 'Pregnant', 'Thyroid_surgery', 'I131_treatment', 
            'Query_hypothyroid', 'Query_hyperthyroid', 'Lithium', 'Goitre',
            'Tumor', 'Hypopituitary', 'Psych'
        ]
        
        numerical_features = ['Age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        label = 'Class'
        
        # Prepare categorical data
        cat_data = df[categorical_features].copy()
        cat_data.columns = [col.lower().replace('_', '_') for col in categorical_features]
        
        # Prepare numerical data
        num_data = df[numerical_features + [label]].copy()
        num_data.columns = [col.lower() for col in numerical_features] + ['class_label']
        
        try:
            # Store categorical data in PostgreSQL
            cat_data.to_sql(
                'thyroid_categorical', 
                self.postgres_engine, 
                if_exists='append', 
                index=False,
                method='multi'
            )
            logger.info(f"✅ Stored {len(cat_data)} categorical records in PostgreSQL")
        except Exception as e:
            logger.error(f"❌ Error storing categorical data: {e}")
            raise
        
        try:
            # Store numerical data in MySQL
            num_data.to_sql(
                'thyroid_numerical', 
                self.mysql_engine, 
                if_exists='append', 
                index=False,
                method='multi'
            )
            logger.info(f"✅ Stored {len(num_data)} numerical records in MySQL")
        except Exception as e:
            logger.error(f"❌ Error storing numerical data: {e}")
            raise
        
        return cat_data, num_data
    
    def verify_and_display_results(self):
        """Verify and display results"""
        print("\n" + "="*70)
        print("🔍 VERIFICATION RESULTS")
        print("="*70)
        
        # Count records
        with self.postgres_engine.connect() as conn:
            pg_count = conn.execute(text("SELECT COUNT(*) FROM thyroid_categorical")).scalar()
            pg_sample = pd.read_sql("SELECT * FROM thyroid_categorical LIMIT 3", conn)
        
        with self.mysql_engine.connect() as conn:
            mysql_count = conn.execute(text("SELECT COUNT(*) FROM thyroid_numerical")).scalar()
            mysql_sample = pd.read_sql("SELECT * FROM thyroid_numerical LIMIT 3", conn)
        
        print(f"📊 PostgreSQL Records: {pg_count}")
        print(f"📊 MySQL Records: {mysql_count}")
        
        if pg_count == mysql_count:
            print("✅ Record counts match!")
        else:
            print("⚠️  Record counts don't match!")
        
        print("\n🗃️  SAMPLE DATA - PostgreSQL (Categorical):")
        print(pg_sample.to_string(index=False))
        
        print("\n🗃️  SAMPLE DATA - MySQL (Numerical):")
        print(mysql_sample.to_string(index=False))

def main():
    """Main execution function"""
    print("🚀 Starting Thyroid Data Splitter...")
    print("="*50)
    
    try:
        # Initialize splitter
        splitter = ThyroidDataSplitter()
        
        # Load data
        df = splitter.load_sample_data()
        print(f"📋 Original data shape: {df.shape}")
        print(f"📋 Features: {list(df.columns)}")
        
        # Create tables
        splitter.create_tables()
        
        # Split and store data
        cat_data, num_data = splitter.split_and_store_data(df)
        
        # Verify results
        splitter.verify_and_display_results()
        
        print("\n✅ SUCCESS! Data splitting completed successfully!")
        print("🎉 Your thyroid dataset has been split and stored in both databases.")
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        print(f"\n❌ FAILED: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()