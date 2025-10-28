import pandas as pd
import os

def csv_to_mysql_inserts(csv_file_path, output_file_path, table_name='thyroid_numeric', batch_size=1000):
    """
    Convert CSV file to MySQL INSERT statements
    
    Args:
        csv_file_path: Path to the CSV file
        output_file_path: Path where to save the SQL file
        table_name: Name of the MySQL table
        batch_size: Number of rows per INSERT statement (for performance)
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    with open(output_file_path, 'w') as f:
        # Write header comments and table creation
        f.write("-- MySQL Script for Complete Thyroid Numeric Data\n")
        f.write(f"-- Generated from {os.path.basename(csv_file_path)}\n")
        f.write(f"-- Total records: {len(df)}\n\n")
        
        # Create table statement
        f.write("-- Create table for thyroid numeric data\n")
        f.write(f"CREATE TABLE {table_name} (\n")
        f.write("    id INT AUTO_INCREMENT PRIMARY KEY,\n")
        f.write("    age DECIMAL(10,4),\n")
        f.write("    TSH DECIMAL(10,6),\n")
        f.write("    T3 DECIMAL(10,4),\n")
        f.write("    TT4 DECIMAL(10,4),\n")
        f.write("    T4U DECIMAL(10,4),\n")
        f.write("    FTI DECIMAL(10,6),\n")
        f.write("    INDEX idx_age (age),\n")
        f.write("    INDEX idx_tsh (TSH)\n")
        f.write(");\n\n")
        
        # Generate INSERT statements in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            f.write(f"-- Batch {i//batch_size + 1} (rows {i+1}-{min(i+batch_size, len(df))})\n")
            f.write(f"INSERT INTO {table_name} (age, TSH, T3, TT4, T4U, FTI) VALUES\n")
            
            values = []
            for _, row in batch.iterrows():
                # Handle potential NaN values
                age = 'NULL' if pd.isna(row['age']) else f"{row['age']}"
                tsh = 'NULL' if pd.isna(row['TSH']) else f"{row['TSH']}"
                t3 = 'NULL' if pd.isna(row['T3']) else f"{row['T3']}"
                tt4 = 'NULL' if pd.isna(row['TT4']) else f"{row['TT4']}"
                t4u = 'NULL' if pd.isna(row['T4U']) else f"{row['T4U']}"
                fti = 'NULL' if pd.isna(row['FTI']) else f"{row['FTI']}"
                
                values.append(f"({age}, {tsh}, {t3}, {tt4}, {t4u}, {fti})")
            
            f.write(',\n'.join(values))
            f.write(';\n\n')
        
        # Add useful queries at the end
        f.write("-- Useful analysis queries:\n\n")
        
        f.write("-- 1. Basic statistics\n")
        f.write("SELECT \n")
        f.write("    COUNT(*) as total_records,\n")
        f.write("    ROUND(AVG(age), 4) as avg_age,\n")
        f.write("    ROUND(MIN(age), 4) as min_age,\n")
        f.write("    ROUND(MAX(age), 4) as max_age,\n")
        f.write("    ROUND(AVG(TSH), 6) as avg_tsh,\n")
        f.write("    ROUND(AVG(T3), 4) as avg_t3,\n")
        f.write("    ROUND(AVG(TT4), 4) as avg_tt4,\n")
        f.write("    ROUND(AVG(T4U), 4) as avg_t4u,\n")
        f.write("    ROUND(AVG(FTI), 6) as avg_fti\n")
        f.write(f"FROM {table_name};\n\n")
        
        f.write("-- 2. Verify data import\n")
        f.write(f"SELECT COUNT(*) as imported_rows FROM {table_name};\n\n")
        
        f.write("-- 3. Sample data check\n")
        f.write(f"SELECT * FROM {table_name} LIMIT 10;\n\n")

if __name__ == "__main__":
    # Set file paths
    csv_file = r"c:\Users\achma\OneDrive\Documents\1Semester 5\PSD\UTS\thyroid_numeric.csv"
    output_file = r"c:\Users\achma\OneDrive\Documents\1Semester 5\PSD\UTS\thyroid_numeric_complete_mysql.sql"
    
    # Generate MySQL INSERT statements
    csv_to_mysql_inserts(csv_file, output_file)
    
    print(f"MySQL script generated successfully!")
    print(f"Output file: {output_file}")
    print("\nTo use this script:")
    print("1. Import into MySQL: mysql -u username -p database_name < thyroid_numeric_complete_mysql.sql")
    print("2. Or copy and paste the SQL commands into your MySQL client")