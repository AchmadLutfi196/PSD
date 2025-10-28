-- PostgreSQL Script for Thyroid Categorical Dataset
-- Comprehensive queries for categorical feature analysis
-- Dataset: thyroid_categorical.csv (7200+ records, 15 features)

-- ===============================
-- TABLE CREATION AND STRUCTURE
-- ===============================

-- Create database (optional)
-- CREATE DATABASE thyroid_categorical_db;
-- \c thyroid_categorical_db;

-- Main categorical table with comprehensive constraints
CREATE TABLE thyroid_categorical (
    id SERIAL PRIMARY KEY,
    sex INTEGER NOT NULL CHECK (sex IN (0, 1)),
    on_thyroxine INTEGER NOT NULL CHECK (on_thyroxine IN (0, 1)),
    query_on_thyroxine INTEGER NOT NULL CHECK (query_on_thyroxine IN (0, 1)),
    on_antithyroid_medication INTEGER NOT NULL CHECK (on_antithyroid_medication IN (0, 1)),
    sick INTEGER NOT NULL CHECK (sick IN (0, 1)),
    pregnant INTEGER NOT NULL CHECK (pregnant IN (0, 1)),
    thyroid_surgery INTEGER NOT NULL CHECK (thyroid_surgery IN (0, 1)),
    i131_treatment INTEGER NOT NULL CHECK (i131_treatment IN (0, 1)),
    query_hypothyroid INTEGER NOT NULL CHECK (query_hypothyroid IN (0, 1)),
    query_hyperthyroid INTEGER NOT NULL CHECK (query_hyperthyroid IN (0, 1)),
    lithium INTEGER NOT NULL CHECK (lithium IN (0, 1)),
    goitre INTEGER NOT NULL CHECK (goitre IN (0, 1)),
    tumor INTEGER NOT NULL CHECK (tumor IN (0, 1)),
    hypopituitary INTEGER NOT NULL CHECK (hypopituitary IN (0, 1)),
    psych INTEGER NOT NULL CHECK (psych IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for frequently queried columns
CREATE INDEX idx_thyroid_sex ON thyroid_categorical(sex);
CREATE INDEX idx_thyroid_on_thyroxine ON thyroid_categorical(on_thyroxine);
CREATE INDEX idx_thyroid_sick ON thyroid_categorical(sick);
CREATE INDEX idx_thyroid_pregnant ON thyroid_categorical(pregnant);
CREATE INDEX idx_thyroid_surgery ON thyroid_categorical(thyroid_surgery);
CREATE INDEX idx_thyroid_treatment ON thyroid_categorical(i131_treatment);

-- Composite indexes for common query patterns
CREATE INDEX idx_thyroid_treatment_combo ON thyroid_categorical(on_thyroxine, on_antithyroid_medication, thyroid_surgery, i131_treatment);
CREATE INDEX idx_thyroid_symptoms ON thyroid_categorical(query_hypothyroid, query_hyperthyroid);

-- ===============================
-- SAMPLE DATA INSERTS (First 20 rows)
-- ===============================

INSERT INTO thyroid_categorical (sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, i131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych) VALUES
(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);

-- Note: For complete dataset (7200+ rows), use the Python generator script

-- ===============================
-- DATA EXPLORATION QUERIES
-- ===============================

-- 1. Basic dataset overview
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT sex) as unique_sex_values,
    MIN(created_at) as first_record,
    MAX(created_at) as last_record
FROM thyroid_categorical;

-- 2. Gender distribution
SELECT 
    CASE 
        WHEN sex = 0 THEN 'Female' 
        ELSE 'Male' 
    END as gender,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM thyroid_categorical
GROUP BY sex
ORDER BY sex;

-- 3. Feature frequency analysis - All categorical features
SELECT 
    'sex' as feature_name,
    sex as feature_value,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM thyroid_categorical
GROUP BY sex

UNION ALL

SELECT 
    'on_thyroxine' as feature_name,
    on_thyroxine as feature_value,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM thyroid_categorical
GROUP BY on_thyroxine

UNION ALL

SELECT 
    'sick' as feature_name,
    sick as feature_value,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM thyroid_categorical
GROUP BY sick

UNION ALL

SELECT 
    'pregnant' as feature_name,
    pregnant as feature_value,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM thyroid_categorical
GROUP BY pregnant

ORDER BY feature_name, feature_value;

-- ===============================
-- MEDICAL CONDITION ANALYSIS
-- ===============================

-- 4. Treatment pattern analysis
SELECT 
    CASE 
        WHEN on_thyroxine = 1 THEN 'On Thyroxine'
        WHEN on_antithyroid_medication = 1 THEN 'On Anti-thyroid Medication'
        WHEN thyroid_surgery = 1 THEN 'Had Thyroid Surgery'
        WHEN i131_treatment = 1 THEN 'Had I131 Treatment'
        ELSE 'No Major Treatment'
    END as treatment_type,
    COUNT(*) as patient_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM thyroid_categorical
GROUP BY 
    CASE 
        WHEN on_thyroxine = 1 THEN 'On Thyroxine'
        WHEN on_antithyroid_medication = 1 THEN 'On Anti-thyroid Medication'
        WHEN thyroid_surgery = 1 THEN 'Had Thyroid Surgery'
        WHEN i131_treatment = 1 THEN 'Had I131 Treatment'
        ELSE 'No Major Treatment'
    END
ORDER BY patient_count DESC;

-- 5. Multiple treatment combinations
SELECT 
    on_thyroxine,
    on_antithyroid_medication,
    thyroid_surgery,
    i131_treatment,
    COUNT(*) as combination_count
FROM thyroid_categorical
WHERE on_thyroxine = 1 OR on_antithyroid_medication = 1 OR thyroid_surgery = 1 OR i131_treatment = 1
GROUP BY on_thyroxine, on_antithyroid_medication, thyroid_surgery, i131_treatment
ORDER BY combination_count DESC;

-- 6. Pregnancy and thyroid conditions
SELECT 
    CASE WHEN sex = 0 THEN 'Female' ELSE 'Male' END as gender,
    pregnant,
    on_thyroxine,
    COUNT(*) as count
FROM thyroid_categorical
WHERE pregnant = 1 OR (sex = 0 AND on_thyroxine = 1)
GROUP BY sex, pregnant, on_thyroxine
ORDER BY count DESC;

-- ===============================
-- SYMPTOM CORRELATION ANALYSIS
-- ===============================

-- 7. Hypothyroid vs Hyperthyroid queries
SELECT 
    CASE 
        WHEN query_hypothyroid = 1 AND query_hyperthyroid = 1 THEN 'Both Hypo & Hyper Query'
        WHEN query_hypothyroid = 1 THEN 'Hypothyroid Query Only'
        WHEN query_hyperthyroid = 1 THEN 'Hyperthyroid Query Only'
        ELSE 'No Thyroid Query'
    END as thyroid_query_type,
    COUNT(*) as patient_count,
    ROUND(AVG(on_thyroxine::NUMERIC) * 100, 1) as pct_on_thyroxine,
    ROUND(AVG(on_antithyroid_medication::NUMERIC) * 100, 1) as pct_on_anti_thyroid
FROM thyroid_categorical
GROUP BY 
    CASE 
        WHEN query_hypothyroid = 1 AND query_hyperthyroid = 1 THEN 'Both Hypo & Hyper Query'
        WHEN query_hypothyroid = 1 THEN 'Hypothyroid Query Only'
        WHEN query_hyperthyroid = 1 THEN 'Hyperthyroid Query Only'
        ELSE 'No Thyroid Query'
    END
ORDER BY patient_count DESC;

-- 8. Physical conditions analysis (goitre, tumor, etc.)
SELECT 
    'Goitre' as condition,
    SUM(goitre) as positive_cases,
    COUNT(*) - SUM(goitre) as negative_cases,
    ROUND(AVG(goitre::NUMERIC) * 100, 2) as prevalence_percentage
FROM thyroid_categorical

UNION ALL

SELECT 
    'Tumor' as condition,
    SUM(tumor) as positive_cases,
    COUNT(*) - SUM(tumor) as negative_cases,
    ROUND(AVG(tumor::NUMERIC) * 100, 2) as prevalence_percentage
FROM thyroid_categorical

UNION ALL

SELECT 
    'Hypopituitary' as condition,
    SUM(hypopituitary) as positive_cases,
    COUNT(*) - SUM(hypopituitary) as negative_cases,
    ROUND(AVG(hypopituitary::NUMERIC) * 100, 2) as prevalence_percentage
FROM thyroid_categorical

ORDER BY prevalence_percentage DESC;

-- ===============================
-- ADVANCED PATTERN ANALYSIS
-- ===============================

-- 9. Co-occurrence matrix for key conditions
SELECT 
    'Sick & On_Thyroxine' as condition_pair,
    SUM(CASE WHEN sick = 1 AND on_thyroxine = 1 THEN 1 ELSE 0 END) as both_present,
    SUM(CASE WHEN sick = 1 AND on_thyroxine = 0 THEN 1 ELSE 0 END) as first_only,
    SUM(CASE WHEN sick = 0 AND on_thyroxine = 1 THEN 1 ELSE 0 END) as second_only,
    SUM(CASE WHEN sick = 0 AND on_thyroxine = 0 THEN 1 ELSE 0 END) as neither
FROM thyroid_categorical

UNION ALL

SELECT 
    'Goitre & Tumor' as condition_pair,
    SUM(CASE WHEN goitre = 1 AND tumor = 1 THEN 1 ELSE 0 END) as both_present,
    SUM(CASE WHEN goitre = 1 AND tumor = 0 THEN 1 ELSE 0 END) as first_only,
    SUM(CASE WHEN goitre = 0 AND tumor = 1 THEN 1 ELSE 0 END) as second_only,
    SUM(CASE WHEN goitre = 0 AND tumor = 0 THEN 1 ELSE 0 END) as neither
FROM thyroid_categorical;

-- 10. Risk profile creation
SELECT 
    id,
    CASE WHEN sex = 0 THEN 'Female' ELSE 'Male' END as gender,
    (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment) as treatment_score,
    (query_hypothyroid + query_hyperthyroid + goitre + tumor + hypopituitary) as symptom_score,
    (sick + lithium + psych) as risk_factor_score,
    CASE 
        WHEN (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment + 
              query_hypothyroid + query_hyperthyroid + goitre + tumor + hypopituitary + 
              sick + lithium + psych) >= 3 THEN 'High Risk'
        WHEN (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment + 
              query_hypothyroid + query_hyperthyroid + goitre + tumor + hypopituitary + 
              sick + lithium + psych) >= 1 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as overall_risk_category
FROM thyroid_categorical
ORDER BY 
    (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment + 
     query_hypothyroid + query_hyperthyroid + goitre + tumor + hypopituitary + 
     sick + lithium + psych) DESC
LIMIT 20;

-- ===============================
-- DATA QUALITY AND VALIDATION
-- ===============================

-- 11. Data completeness check
SELECT 
    'Total Records' as metric,
    COUNT(*) as value
FROM thyroid_categorical

UNION ALL

SELECT 
    'Records with any NULL' as metric,
    COUNT(*) as value
FROM thyroid_categorical
WHERE sex IS NULL OR on_thyroxine IS NULL OR query_on_thyroxine IS NULL OR 
      on_antithyroid_medication IS NULL OR sick IS NULL OR pregnant IS NULL OR
      thyroid_surgery IS NULL OR i131_treatment IS NULL OR query_hypothyroid IS NULL OR
      query_hyperthyroid IS NULL OR lithium IS NULL OR goitre IS NULL OR
      tumor IS NULL OR hypopituitary IS NULL OR psych IS NULL

UNION ALL

SELECT 
    'Complete Records' as metric,
    COUNT(*) as value
FROM thyroid_categorical
WHERE sex IS NOT NULL AND on_thyroxine IS NOT NULL AND query_on_thyroxine IS NOT NULL AND 
      on_antithyroid_medication IS NOT NULL AND sick IS NOT NULL AND pregnant IS NOT NULL AND
      thyroid_surgery IS NOT NULL AND i131_treatment IS NOT NULL AND query_hypothyroid IS NOT NULL AND
      query_hyperthyroid IS NOT NULL AND lithium IS NOT NULL AND goitre IS NOT NULL AND
      tumor IS NOT NULL AND hypopituitary IS NOT NULL AND psych IS NOT NULL;

-- 12. Feature summary statistics
WITH feature_stats AS (
    SELECT 
        'sex' as feature, AVG(sex::NUMERIC) as mean_value, COUNT(DISTINCT sex) as unique_values
    FROM thyroid_categorical
    UNION ALL
    SELECT 
        'on_thyroxine', AVG(on_thyroxine::NUMERIC), COUNT(DISTINCT on_thyroxine)
    FROM thyroid_categorical
    UNION ALL
    SELECT 
        'pregnant', AVG(pregnant::NUMERIC), COUNT(DISTINCT pregnant)
    FROM thyroid_categorical
    UNION ALL
    SELECT 
        'sick', AVG(sick::NUMERIC), COUNT(DISTINCT sick)
    FROM thyroid_categorical
    UNION ALL
    SELECT 
        'goitre', AVG(goitre::NUMERIC), COUNT(DISTINCT goitre)
    FROM thyroid_categorical
)
SELECT 
    feature,
    ROUND(mean_value, 4) as prevalence_rate,
    unique_values,
    CASE 
        WHEN unique_values = 2 THEN 'Binary Feature'
        ELSE 'Multi-valued Feature'
    END as feature_type
FROM feature_stats
ORDER BY prevalence_rate DESC;

-- ===============================
-- UTILITY VIEWS AND FUNCTIONS
-- ===============================

-- Create view for easy analysis
CREATE OR REPLACE VIEW thyroid_summary_view AS
SELECT 
    id,
    CASE WHEN sex = 0 THEN 'Female' ELSE 'Male' END as gender,
    CASE WHEN on_thyroxine = 1 THEN 'Yes' ELSE 'No' END as taking_thyroxine,
    CASE WHEN pregnant = 1 THEN 'Yes' ELSE 'No' END as is_pregnant,
    CASE WHEN sick = 1 THEN 'Yes' ELSE 'No' END as is_sick,
    (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment) as total_treatments,
    (query_hypothyroid + query_hyperthyroid + goitre + tumor + hypopituitary + sick + lithium + psych) as total_conditions,
    created_at
FROM thyroid_categorical;

-- Create materialized view for performance (optional)
-- CREATE MATERIALIZED VIEW thyroid_analytics AS
-- SELECT * FROM thyroid_summary_view;

-- ===============================
-- SAMPLE QUERIES FOR ANALYSIS
-- ===============================

-- Query 1: Find patients with multiple treatments
SELECT * FROM thyroid_summary_view WHERE total_treatments > 1;

-- Query 2: Gender-based treatment analysis
SELECT 
    gender,
    AVG(total_treatments) as avg_treatments,
    COUNT(*) as patient_count
FROM thyroid_summary_view
GROUP BY gender;

-- Query 3: High-risk patients identification
SELECT * 
FROM thyroid_summary_view 
WHERE total_conditions >= 2 
ORDER BY total_conditions DESC, total_treatments DESC;

-- ===============================
-- PERFORMANCE OPTIMIZATION
-- ===============================

-- Update table statistics for query optimization
ANALYZE thyroid_categorical;

-- Example of adding a computed column (optional)
-- ALTER TABLE thyroid_categorical 
-- ADD COLUMN risk_score INTEGER GENERATED ALWAYS AS 
-- (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment + 
--  query_hypothyroid + query_hyperthyroid + goitre + tumor + hypopituitary + 
--  sick + lithium + psych) STORED;

-- Final verification
SELECT 'Data Import Verification' as status, COUNT(*) as record_count 
FROM thyroid_categorical;