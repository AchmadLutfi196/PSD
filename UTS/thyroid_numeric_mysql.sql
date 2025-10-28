-- MySQL Script for Thyroid Numeric Data
-- Created from thyroid_numeric.csv

-- Create database (optional)
-- CREATE DATABASE thyroid_data;
-- USE thyroid_data;

-- Create table for thyroid numeric data
CREATE TABLE thyroid_numeric (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age DECIMAL(10,4),
    TSH DECIMAL(10,6),
    T3 DECIMAL(10,4),
    TT4 DECIMAL(10,4),
    T4U DECIMAL(10,4),
    FTI DECIMAL(10,6),
    INDEX idx_age (age),
    INDEX idx_tsh (TSH)
);

-- Insert data statements
INSERT INTO thyroid_numeric (age, TSH, T3, TT4, T4U, FTI) VALUES
(0.73, 0.0006, 0.015, 0.12, 0.082, 0.146),
(0.24, 0.00025, 0.03, 0.143, 0.133, 0.108),
(0.47, 0.0019, 0.024, 0.102, 0.131, 0.078),
(0.64, 0.0009, 0.017, 0.077, 0.09, 0.085),
(0.23, 0.00025, 0.026, 0.139, 0.09, 0.153),
(0.69, 0.00025, 0.016, 0.086, 0.07, 0.123),
(0.85, 0.00025, 0.023, 0.128, 0.104, 0.121),
(0.48, 0.00208, 0.02, 0.086, 0.078, 0.11),
(0.67, 0.0013, 0.024, 0.087, 0.109, 0.08),
(0.76, 0.0001, 0.029, 0.124, 0.128, 0.097),
(0.62, 0.011, 0.008, 0.073, 0.074, 0.098),
(0.18, 0.0001, 0.023, 0.098, 0.085, 0.115),
(0.59, 0.0008, 0.023, 0.094, 0.099, 0.09475),
(0.49, 0.0006, 0.023, 0.113, 0.102, 0.111),
(0.53, 0.0023, 0.02, 0.063, 0.095, 0.066),
(0.39, 0.0001, 0.018, 0.09, 0.071, 0.126),
(0.39, 0.0006, 0.02, 0.114, 0.1, 0.114),
(0.65, 0.0016, 0.018, 0.078, 0.092, 0.085),
(0.64, 0.032, 0.014, 0.085, 0.116, 0.071);

-- NOTE: This is just a sample of the first 19 rows.
-- For the complete dataset (7200+ rows), you would need to process the entire CSV file.

-- Useful queries for analysis:

-- 1. Basic statistics
SELECT 
    COUNT(*) as total_records,
    AVG(age) as avg_age,
    MIN(age) as min_age,
    MAX(age) as max_age,
    AVG(TSH) as avg_tsh,
    AVG(T3) as avg_t3,
    AVG(TT4) as avg_tt4,
    AVG(T4U) as avg_t4u,
    AVG(FTI) as avg_fti
FROM thyroid_numeric;

-- 2. Age distribution
SELECT 
    CASE 
        WHEN age < 0.3 THEN 'Young (< 0.3)'
        WHEN age BETWEEN 0.3 AND 0.6 THEN 'Middle (0.3-0.6)'
        ELSE 'Older (> 0.6)'
    END as age_group,
    COUNT(*) as count,
    AVG(TSH) as avg_tsh
FROM thyroid_numeric
GROUP BY 
    CASE 
        WHEN age < 0.3 THEN 'Young (< 0.3)'
        WHEN age BETWEEN 0.3 AND 0.6 THEN 'Middle (0.3-0.6)'
        ELSE 'Older (> 0.6)'
    END;

-- 3. Find outliers (TSH values)
SELECT *
FROM thyroid_numeric
WHERE TSH > (SELECT AVG(TSH) + 2 * STD(TSH) FROM thyroid_numeric)
   OR TSH < (SELECT AVG(TSH) - 2 * STD(TSH) FROM thyroid_numeric);

-- 4. Correlation analysis preparation
SELECT 
    age,
    TSH,
    T3,
    TT4,
    T4U,
    FTI,
    (age - (SELECT AVG(age) FROM thyroid_numeric)) * 
    (TSH - (SELECT AVG(TSH) FROM thyroid_numeric)) as age_tsh_product
FROM thyroid_numeric;

-- 5. Top 10 highest TSH values
SELECT *
FROM thyroid_numeric
ORDER BY TSH DESC
LIMIT 10;

-- 6. Data validation - check for null values
SELECT 
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as null_age,
    SUM(CASE WHEN TSH IS NULL THEN 1 ELSE 0 END) as null_tsh,
    SUM(CASE WHEN T3 IS NULL THEN 1 ELSE 0 END) as null_t3,
    SUM(CASE WHEN TT4 IS NULL THEN 1 ELSE 0 END) as null_tt4,
    SUM(CASE WHEN T4U IS NULL THEN 1 ELSE 0 END) as null_t4u,
    SUM(CASE WHEN FTI IS NULL THEN 1 ELSE 0 END) as null_fti
FROM thyroid_numeric;