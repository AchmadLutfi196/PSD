import pandas as pd
import os

def csv_to_postgresql_inserts(csv_files_dict, output_file_path, batch_size=1000):
    """
    Convert multiple CSV files to PostgreSQL INSERT statements
    
    Args:
        csv_files_dict: Dictionary with table_name: csv_file_path pairs
        output_file_path: Path where to save the SQL file
        batch_size: Number of rows per INSERT statement (for performance)
    """
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("-- PostgreSQL Script for Complete Thyroid Dataset\n")
        f.write("-- Generated from multiple CSV files\n")
        f.write(f"-- Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Create tables first
        f.write("-- Create tables\n\n")
        
        # Thyroid numeric table
        f.write("CREATE TABLE thyroid_numeric (\n")
        f.write("    id SERIAL PRIMARY KEY,\n")
        f.write("    age DECIMAL(10,4),\n")
        f.write("    tsh DECIMAL(10,6),\n")
        f.write("    t3 DECIMAL(10,4),\n")
        f.write("    tt4 DECIMAL(10,4),\n")
        f.write("    t4u DECIMAL(10,4),\n")
        f.write("    fti DECIMAL(10,6),\n")
        f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n")
        f.write(");\n\n")
        
        # Thyroid categorical table
        f.write("CREATE TABLE thyroid_categorical (\n")
        f.write("    id SERIAL PRIMARY KEY,\n")
        f.write("    sex INTEGER CHECK (sex IN (0, 1)),\n")
        f.write("    on_thyroxine INTEGER CHECK (on_thyroxine IN (0, 1)),\n")
        f.write("    query_on_thyroxine INTEGER CHECK (query_on_thyroxine IN (0, 1)),\n")
        f.write("    on_antithyroid_medication INTEGER CHECK (on_antithyroid_medication IN (0, 1)),\n")
        f.write("    sick INTEGER CHECK (sick IN (0, 1)),\n")
        f.write("    pregnant INTEGER CHECK (pregnant IN (0, 1)),\n")
        f.write("    thyroid_surgery INTEGER CHECK (thyroid_surgery IN (0, 1)),\n")
        f.write("    i131_treatment INTEGER CHECK (i131_treatment IN (0, 1)),\n")
        f.write("    query_hypothyroid INTEGER CHECK (query_hypothyroid IN (0, 1)),\n")
        f.write("    query_hyperthyroid INTEGER CHECK (query_hyperthyroid IN (0, 1)),\n")
        f.write("    lithium INTEGER CHECK (lithium IN (0, 1)),\n")
        f.write("    goitre INTEGER CHECK (goitre IN (0, 1)),\n")
        f.write("    tumor INTEGER CHECK (tumor IN (0, 1)),\n")
        f.write("    hypopituitary INTEGER CHECK (hypopituitary IN (0, 1)),\n")
        f.write("    psych INTEGER CHECK (psych IN (0, 1)),\n")
        f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n")
        f.write(");\n\n")
        
        # Thyroid labels table
        f.write("CREATE TABLE thyroid_labels (\n")
        f.write("    id SERIAL PRIMARY KEY,\n")
        f.write("    class INTEGER NOT NULL,\n")
        f.write("    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n")
        f.write(");\n\n")
        
        # Create indexes
        f.write("-- Create indexes for better performance\n")
        f.write("CREATE INDEX idx_thyroid_numeric_age ON thyroid_numeric(age);\n")
        f.write("CREATE INDEX idx_thyroid_numeric_tsh ON thyroid_numeric(tsh);\n")
        f.write("CREATE INDEX idx_thyroid_categorical_sex ON thyroid_categorical(sex);\n")
        f.write("CREATE INDEX idx_thyroid_labels_class ON thyroid_labels(class);\n\n")
        
        # Process each CSV file
        for table_name, csv_file_path in csv_files_dict.items():
            if not os.path.exists(csv_file_path):
                print(f"Warning: File {csv_file_path} not found, skipping...")
                continue
                
            df = pd.read_csv(csv_file_path)
            f.write(f"-- Data for {table_name} (Total records: {len(df)})\n")
            
            # Generate INSERT statements in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                f.write(f"-- Batch {i//batch_size + 1} (rows {i+1}-{min(i+batch_size, len(df))})\n")
                
                if table_name == 'thyroid_numeric':
                    f.write("INSERT INTO thyroid_numeric (age, tsh, t3, tt4, t4u, fti) VALUES\n")
                    values = []
                    for _, row in batch.iterrows():
                        age = 'NULL' if pd.isna(row['age']) else f"{row['age']}"
                        tsh = 'NULL' if pd.isna(row['TSH']) else f"{row['TSH']}"
                        t3 = 'NULL' if pd.isna(row['T3']) else f"{row['T3']}"
                        tt4 = 'NULL' if pd.isna(row['TT4']) else f"{row['TT4']}"
                        t4u = 'NULL' if pd.isna(row['T4U']) else f"{row['T4U']}"
                        fti = 'NULL' if pd.isna(row['FTI']) else f"{row['FTI']}"
                        values.append(f"({age}, {tsh}, {t3}, {tt4}, {t4u}, {fti})")
                
                elif table_name == 'thyroid_categorical':
                    f.write("INSERT INTO thyroid_categorical (sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, i131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych) VALUES\n")
                    values = []
                    for _, row in batch.iterrows():
                        values.append(f"({row['sex']}, {row['on_thyroxine']}, {row['query_on_thyroxine']}, {row['on_antithyroid_medication']}, {row['sick']}, {row['pregnant']}, {row['thyroid_surgery']}, {row['I131_treatment']}, {row['query_hypothyroid']}, {row['query_hyperthyroid']}, {row['lithium']}, {row['goitre']}, {row['tumor']}, {row['hypopituitary']}, {row['psych']})")
                
                elif table_name == 'thyroid_labels':
                    f.write("INSERT INTO thyroid_labels (class) VALUES\n")
                    values = []
                    for _, row in batch.iterrows():
                        values.append(f"({row['class']})")
                
                f.write(',\n'.join(values))
                f.write(';\n\n')
        
        # Create combined view
        f.write("-- Create combined view for analysis\n")
        f.write("CREATE VIEW thyroid_complete AS\n")
        f.write("SELECT \n")
        f.write("    tn.id,\n")
        f.write("    tn.age, tn.tsh, tn.t3, tn.tt4, tn.t4u, tn.fti,\n")
        f.write("    tc.sex, tc.on_thyroxine, tc.query_on_thyroxine, tc.on_antithyroid_medication,\n")
        f.write("    tc.sick, tc.pregnant, tc.thyroid_surgery, tc.i131_treatment,\n")
        f.write("    tc.query_hypothyroid, tc.query_hyperthyroid, tc.lithium,\n")
        f.write("    tc.goitre, tc.tumor, tc.hypopituitary, tc.psych,\n")
        f.write("    tl.class\n")
        f.write("FROM thyroid_numeric tn\n")
        f.write("JOIN thyroid_categorical tc ON tn.id = tc.id\n")
        f.write("JOIN thyroid_labels tl ON tn.id = tl.id;\n\n")
        
        # Add verification queries
        f.write("-- Verification queries\n")
        f.write("SELECT 'thyroid_numeric' as table_name, COUNT(*) as row_count FROM thyroid_numeric\n")
        f.write("UNION ALL\n")
        f.write("SELECT 'thyroid_categorical' as table_name, COUNT(*) as row_count FROM thyroid_categorical\n")
        f.write("UNION ALL\n")
        f.write("SELECT 'thyroid_labels' as table_name, COUNT(*) as row_count FROM thyroid_labels;\n\n")
        
        f.write("-- Sample data check\n")
        f.write("SELECT * FROM thyroid_complete LIMIT 10;\n\n")
        
        f.write("-- Basic statistics\n")
        f.write("SELECT \n")
        f.write("    COUNT(*) as total_records,\n")
        f.write("    COUNT(DISTINCT class) as unique_classes,\n")
        f.write("    ROUND(AVG(age), 4) as avg_age,\n")
        f.write("    ROUND(AVG(tsh), 6) as avg_tsh\n")
        f.write("FROM thyroid_complete;\n")

if __name__ == "__main__":
    # Set file paths
    base_path = r"c:\Users\achma\OneDrive\Documents\1Semester 5\PSD\UTS"
    
    csv_files = {
        'thyroid_numeric': os.path.join(base_path, 'thyroid_numeric.csv'),
        'thyroid_categorical': os.path.join(base_path, 'thyroid_categorical.csv'),
        'thyroid_labels': os.path.join(base_path, 'thyroid_label.csv')
    }
    
    output_file = os.path.join(base_path, 'thyroid_postgresql_complete.sql')
    
    # Generate PostgreSQL INSERT statements
    csv_to_postgresql_inserts(csv_files, output_file, batch_size=500)
    
    print("PostgreSQL script generated successfully!")
    print(f"Output file: {output_file}")
    print(f"\nDataset summary:")
    for table_name, csv_file in csv_files.items():
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"- {table_name}: {len(df)} rows, {len(df.columns)} columns")
    
    print("\nTo use this script:")
    print("1. Create database: createdb thyroid_data")
    print("2. Import data: psql -d thyroid_data -f thyroid_postgresql_complete.sql")
    print("3. Or use pgAdmin to execute the SQL script")