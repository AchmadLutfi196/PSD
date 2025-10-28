import pandas as pd
import os

def categorical_csv_to_postgresql(csv_file_path, output_file_path, batch_size=500):
    """
    Convert thyroid categorical CSV to PostgreSQL INSERT statements
    
    Args:
        csv_file_path: Path to the thyroid_categorical.csv file
        output_file_path: Path where to save the SQL file
        batch_size: Number of rows per INSERT statement (for performance)
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get column information
    columns = df.columns.tolist()
    print(f"Processing {len(df)} rows with columns: {columns}")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # Write header comments
        f.write("-- PostgreSQL Script for Thyroid Categorical Dataset\n")
        f.write(f"-- Generated from {os.path.basename(csv_file_path)}\n")
        f.write(f"-- Total records: {len(df)}\n")
        f.write(f"-- Features: {len(columns)}\n")
        f.write(f"-- Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Table creation with constraints
        f.write("-- Create thyroid categorical table with constraints\n")
        f.write("DROP TABLE IF EXISTS thyroid_categorical CASCADE;\n\n")
        f.write("CREATE TABLE thyroid_categorical (\n")
        f.write("    id SERIAL PRIMARY KEY,\n")
        
        # Add each column with appropriate constraints
        for col in columns:
            col_clean = col.lower().replace('i131_treatment', 'i131_treatment')
            f.write(f"    {col_clean} INTEGER NOT NULL CHECK ({col_clean} IN (0, 1)),\n")
        
        f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n")
        f.write("    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n")
        f.write(");\n\n")
        
        # Create indexes
        f.write("-- Create indexes for better query performance\n")
        important_columns = ['sex', 'on_thyroxine', 'sick', 'pregnant', 'thyroid_surgery', 'i131_treatment']
        
        for col in important_columns:
            if col in [c.lower() for c in columns]:
                f.write(f"CREATE INDEX idx_thyroid_{col} ON thyroid_categorical({col});\n")
        
        # Composite indexes
        f.write("CREATE INDEX idx_thyroid_treatment_combo ON thyroid_categorical(on_thyroxine, on_antithyroid_medication, thyroid_surgery, i131_treatment);\n")
        f.write("CREATE INDEX idx_thyroid_symptoms ON thyroid_categorical(query_hypothyroid, query_hyperthyroid);\n\n")
        
        # Data validation and summary before inserts
        f.write("-- Data validation summary\n")
        f.write("-- Expected data characteristics:\n")
        for col in columns:
            unique_vals = df[col].nunique()
            value_counts = df[col].value_counts().to_dict()
            f.write(f"-- {col}: {unique_vals} unique values, distribution: {value_counts}\n")
        f.write("\n")
        
        # Generate INSERT statements in batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        f.write(f"-- Inserting data in {total_batches} batches of {batch_size} records each\n\n")
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            f.write(f"-- Batch {batch_num}/{total_batches} (rows {i+1}-{min(i+batch_size, len(df))})\n")
            
            # Create column list (mapping CSV headers to database columns)
            db_columns = []
            for col in columns:
                if col == 'I131_treatment':
                    db_columns.append('i131_treatment')
                else:
                    db_columns.append(col.lower())
            
            column_list = ', '.join(db_columns)
            f.write(f"INSERT INTO thyroid_categorical ({column_list}) VALUES\n")
            
            values = []
            for _, row in batch.iterrows():
                row_values = []
                for col in columns:
                    val = row[col]
                    # Handle potential NaN values (though shouldn't exist in categorical data)
                    if pd.isna(val):
                        row_values.append('NULL')
                    else:
                        row_values.append(str(int(val)))  # Ensure integer format
                
                values.append(f"({', '.join(row_values)})")
            
            f.write(',\n'.join(values))
            f.write(';\n\n')
        
        # Create useful views
        f.write("-- Create summary view for easy analysis\n")
        f.write("CREATE OR REPLACE VIEW thyroid_categorical_summary AS\n")
        f.write("SELECT \n")
        f.write("    id,\n")
        f.write("    CASE WHEN sex = 0 THEN 'Female' ELSE 'Male' END as gender,\n")
        f.write("    CASE WHEN on_thyroxine = 1 THEN 'Yes' ELSE 'No' END as taking_thyroxine,\n")
        f.write("    CASE WHEN pregnant = 1 THEN 'Yes' ELSE 'No' END as is_pregnant,\n")
        f.write("    CASE WHEN sick = 1 THEN 'Yes' ELSE 'No' END as is_sick,\n")
        f.write("    (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment) as treatment_count,\n")
        f.write("    (query_hypothyroid + query_hyperthyroid + goitre + tumor + hypopituitary) as condition_count,\n")
        f.write("    created_at\n")
        f.write("FROM thyroid_categorical;\n\n")
        
        # Add verification and analysis queries
        f.write("-- Verification queries\n")
        f.write("SELECT 'Total Records Imported' as metric, COUNT(*) as value FROM thyroid_categorical;\n\n")
        
        f.write("-- Basic statistics\n")
        f.write("SELECT \n")
        f.write("    'Gender Distribution' as analysis,\n")
        f.write("    CASE WHEN sex = 0 THEN 'Female' ELSE 'Male' END as category,\n")
        f.write("    COUNT(*) as count,\n")
        f.write("    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage\n")
        f.write("FROM thyroid_categorical\n")
        f.write("GROUP BY sex\n")
        f.write("ORDER BY count DESC;\n\n")
        
        f.write("-- Treatment prevalence\n")
        f.write("WITH treatment_stats AS (\n")
        f.write("    SELECT \n")
        f.write("        'On Thyroxine' as treatment, SUM(on_thyroxine) as count\n")
        f.write("    FROM thyroid_categorical\n")
        f.write("    UNION ALL\n")
        f.write("    SELECT \n")
        f.write("        'Anti-thyroid Medication', SUM(on_antithyroid_medication)\n")
        f.write("    FROM thyroid_categorical\n")
        f.write("    UNION ALL\n")
        f.write("    SELECT \n")
        f.write("        'Thyroid Surgery', SUM(thyroid_surgery)\n")
        f.write("    FROM thyroid_categorical\n")
        f.write("    UNION ALL\n")
        f.write("    SELECT \n")
        f.write("        'I131 Treatment', SUM(i131_treatment)\n")
        f.write("    FROM thyroid_categorical\n")
        f.write(")\n")
        f.write("SELECT \n")
        f.write("    treatment,\n")
        f.write("    count as patient_count,\n")
        f.write("    ROUND(count * 100.0 / (SELECT COUNT(*) FROM thyroid_categorical), 2) as prevalence_pct\n")
        f.write("FROM treatment_stats\n")
        f.write("ORDER BY patient_count DESC;\n\n")
        
        f.write("-- Sample data verification\n")
        f.write("SELECT * FROM thyroid_categorical_summary LIMIT 10;\n\n")
        
        f.write("-- Data quality check\n")
        f.write("SELECT \n")
        f.write("    COUNT(*) as total_records,\n")
        f.write("    COUNT(CASE WHEN sex IN (0,1) THEN 1 END) as valid_sex,\n")
        f.write("    COUNT(CASE WHEN on_thyroxine IN (0,1) THEN 1 END) as valid_thyroxine,\n")
        f.write("    COUNT(CASE WHEN pregnant IN (0,1) THEN 1 END) as valid_pregnant\n")
        f.write("FROM thyroid_categorical;\n")

if __name__ == "__main__":
    # Set file paths
    base_path = r"c:\Users\achma\OneDrive\Documents\1Semester 5\PSD\UTS"
    csv_file = os.path.join(base_path, 'thyroid_categorical.csv')
    output_file = os.path.join(base_path, 'thyroid_categorical_complete_postgresql.sql')
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        exit(1)
    
    # Generate PostgreSQL INSERT statements
    categorical_csv_to_postgresql(csv_file, output_file, batch_size=500)
    
    # Read and display summary
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*60}")
    print("POSTGRESQL SCRIPT GENERATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Input file: {csv_file}")
    print(f"Output file: {output_file}")
    print(f"\nDataset Summary:")
    print(f"- Total records: {len(df):,}")
    print(f"- Features: {len(df.columns)}")
    print(f"- Feature names: {', '.join(df.columns.tolist())}")
    
    print(f"\nFeature Statistics:")
    for col in df.columns:
        print(f"- {col}: {df[col].nunique()} unique values {dict(df[col].value_counts())}")
    
    print(f"\nTo import into PostgreSQL:")
    print("1. Create database: createdb thyroid_categorical_db")
    print("2. Import: psql -d thyroid_categorical_db -f thyroid_categorical_complete_postgresql.sql")
    print("3. Or use pgAdmin to execute the script")
    print(f"\nQueries generated include:")
    print("- Data validation and constraints")
    print("- Performance indexes")
    print("- Summary views")
    print("- Statistical analysis queries")