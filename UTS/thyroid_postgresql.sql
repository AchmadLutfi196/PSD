-- PostgreSQL Script for Complete Thyroid Dataset
-- Created from thyroid CSV files: numeric, categorical, and label data

-- Create database (optional)
-- CREATE DATABASE thyroid_data;
-- \c thyroid_data;

-- Create table for thyroid numeric data
CREATE TABLE thyroid_numeric (
    id SERIAL PRIMARY KEY,
    age DECIMAL(10,4),
    tsh DECIMAL(10,6),
    t3 DECIMAL(10,4),
    tt4 DECIMAL(10,4),
    t4u DECIMAL(10,4),
    fti DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_thyroid_numeric_age ON thyroid_numeric(age);
CREATE INDEX idx_thyroid_numeric_tsh ON thyroid_numeric(tsh);

-- Create table for thyroid categorical data
CREATE TABLE thyroid_categorical (
    id SERIAL PRIMARY KEY,
    sex INTEGER CHECK (sex IN (0, 1)),
    on_thyroxine INTEGER CHECK (on_thyroxine IN (0, 1)),
    query_on_thyroxine INTEGER CHECK (query_on_thyroxine IN (0, 1)),
    on_antithyroid_medication INTEGER CHECK (on_antithyroid_medication IN (0, 1)),
    sick INTEGER CHECK (sick IN (0, 1)),
    pregnant INTEGER CHECK (pregnant IN (0, 1)),
    thyroid_surgery INTEGER CHECK (thyroid_surgery IN (0, 1)),
    i131_treatment INTEGER CHECK (i131_treatment IN (0, 1)),
    query_hypothyroid INTEGER CHECK (query_hypothyroid IN (0, 1)),
    query_hyperthyroid INTEGER CHECK (query_hyperthyroid IN (0, 1)),
    lithium INTEGER CHECK (lithium IN (0, 1)),
    goitre INTEGER CHECK (goitre IN (0, 1)),
    tumor INTEGER CHECK (tumor IN (0, 1)),
    hypopituitary INTEGER CHECK (hypopituitary IN (0, 1)),
    psych INTEGER CHECK (psych IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for categorical data
CREATE INDEX idx_thyroid_categorical_sex ON thyroid_categorical(sex);
CREATE INDEX idx_thyroid_categorical_on_thyroxine ON thyroid_categorical(on_thyroxine);

-- Create table for thyroid labels/classes
CREATE TABLE thyroid_labels (
    id SERIAL PRIMARY KEY,
    class INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for labels
CREATE INDEX idx_thyroid_labels_class ON thyroid_labels(class);

-- Create a combined view for analysis
CREATE VIEW thyroid_complete AS
SELECT 
    tn.id,
    tn.age,
    tn.tsh,
    tn.t3,
    tn.tt4,
    tn.t4u,
    tn.fti,
    tc.sex,
    tc.on_thyroxine,
    tc.query_on_thyroxine,
    tc.on_antithyroid_medication,
    tc.sick,
    tc.pregnant,
    tc.thyroid_surgery,
    tc.i131_treatment,
    tc.query_hypothyroid,
    tc.query_hyperthyroid,
    tc.lithium,
    tc.goitre,
    tc.tumor,
    tc.hypopituitary,
    tc.psych,
    tl.class
FROM thyroid_numeric tn
JOIN thyroid_categorical tc ON tn.id = tc.id
JOIN thyroid_labels tl ON tn.id = tl.id;

-- Sample INSERT statements (first few rows)
-- For complete data, use the Python script to generate all inserts

-- Sample numeric data
INSERT INTO thyroid_numeric (age, tsh, t3, tt4, t4u, fti) VALUES
(0.73, 0.0006, 0.015, 0.12, 0.082, 0.146),
(0.24, 0.00025, 0.03, 0.143, 0.133, 0.108),
(0.47, 0.0019, 0.024, 0.102, 0.131, 0.078),
(0.64, 0.0009, 0.017, 0.077, 0.09, 0.085),
(0.23, 0.00025, 0.026, 0.139, 0.09, 0.153);

-- Sample categorical data
INSERT INTO thyroid_categorical (sex, on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, i131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych) VALUES
(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

-- Sample label data
INSERT INTO thyroid_labels (class) VALUES
(3), (3), (3), (3), (3);

-- USEFUL ANALYSIS QUERIES FOR POSTGRESQL

-- 1. Basic statistics
SELECT 
    COUNT(*) as total_records,
    ROUND(AVG(age), 4) as avg_age,
    ROUND(MIN(age), 4) as min_age,
    ROUND(MAX(age), 4) as max_age,
    ROUND(AVG(tsh), 6) as avg_tsh,
    ROUND(AVG(t3), 4) as avg_t3,
    ROUND(AVG(tt4), 4) as avg_tt4,
    ROUND(AVG(t4u), 4) as avg_t4u,
    ROUND(AVG(fti), 6) as avg_fti
FROM thyroid_numeric;

-- 2. Class distribution
SELECT 
    class,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM thyroid_labels
GROUP BY class
ORDER BY class;

-- 3. Gender distribution by class
SELECT 
    tl.class,
    CASE WHEN tc.sex = 0 THEN 'Female' ELSE 'Male' END as gender,
    COUNT(*) as count
FROM thyroid_labels tl
JOIN thyroid_categorical tc ON tl.id = tc.id
GROUP BY tl.class, tc.sex
ORDER BY tl.class, tc.sex;

-- 4. Age statistics by class
SELECT 
    tl.class,
    COUNT(*) as count,
    ROUND(AVG(tn.age), 4) as avg_age,
    ROUND(MIN(tn.age), 4) as min_age,
    ROUND(MAX(tn.age), 4) as max_age,
    ROUND(STDDEV(tn.age), 4) as std_age
FROM thyroid_labels tl
JOIN thyroid_numeric tn ON tl.id = tn.id
GROUP BY tl.class
ORDER BY tl.class;

-- 5. Thyroid treatment patterns
SELECT 
    on_thyroxine,
    on_antithyroid_medication,
    thyroid_surgery,
    i131_treatment,
    COUNT(*) as count
FROM thyroid_categorical
WHERE on_thyroxine = 1 OR on_antithyroid_medication = 1 OR thyroid_surgery = 1 OR i131_treatment = 1
GROUP BY on_thyroxine, on_antithyroid_medication, thyroid_surgery, i131_treatment
ORDER BY count DESC;

-- 6. TSH outliers detection using IQR method
WITH tsh_stats AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY tsh) as q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tsh) as q3
    FROM thyroid_numeric
),
tsh_outliers AS (
    SELECT 
        tn.*,
        ts.q1,
        ts.q3,
        ts.q3 - ts.q1 as iqr
    FROM thyroid_numeric tn
    CROSS JOIN tsh_stats ts
    WHERE tn.tsh < (ts.q1 - 1.5 * (ts.q3 - ts.q1)) 
       OR tn.tsh > (ts.q3 + 1.5 * (ts.q3 - ts.q1))
)
SELECT 
    id, age, tsh, t3, tt4, t4u, fti,
    CASE 
        WHEN tsh < (q1 - 1.5 * iqr) THEN 'Low Outlier'
        ELSE 'High Outlier'
    END as outlier_type
FROM tsh_outliers
ORDER BY tsh;

-- 7. Correlation analysis (TSH vs other numeric variables)
SELECT 
    CORR(tsh, age) as tsh_age_correlation,
    CORR(tsh, t3) as tsh_t3_correlation,
    CORR(tsh, tt4) as tsh_tt4_correlation,
    CORR(tsh, t4u) as tsh_t4u_correlation,
    CORR(tsh, fti) as tsh_fti_correlation
FROM thyroid_numeric;

-- 8. Feature importance by class (categorical features)
SELECT 
    'on_thyroxine' as feature,
    class,
    ROUND(AVG(on_thyroxine::NUMERIC), 3) as avg_value
FROM thyroid_complete
GROUP BY class
UNION ALL
SELECT 
    'sick' as feature,
    class,
    ROUND(AVG(sick::NUMERIC), 3) as avg_value
FROM thyroid_complete
GROUP BY class
UNION ALL
SELECT 
    'pregnant' as feature,
    class,
    ROUND(AVG(pregnant::NUMERIC), 3) as avg_value
FROM thyroid_complete
GROUP BY class
ORDER BY feature, class;

-- 9. Complex analysis: Age groups vs TSH levels vs Class
SELECT 
    CASE 
        WHEN age < 0.3 THEN 'Young (< 0.3)'
        WHEN age BETWEEN 0.3 AND 0.6 THEN 'Middle (0.3-0.6)'
        ELSE 'Older (> 0.6)'
    END as age_group,
    class,
    COUNT(*) as count,
    ROUND(AVG(tsh), 6) as avg_tsh,
    ROUND(MIN(tsh), 6) as min_tsh,
    ROUND(MAX(tsh), 6) as max_tsh
FROM thyroid_complete
GROUP BY 
    CASE 
        WHEN age < 0.3 THEN 'Young (< 0.3)'
        WHEN age BETWEEN 0.3 AND 0.6 THEN 'Middle (0.3-0.6)'
        ELSE 'Older (> 0.6)'
    END,
    class
ORDER BY age_group, class;

-- 10. Data quality check
SELECT 
    'thyroid_numeric' as table_name,
    COUNT(*) as total_rows,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as null_age,
    SUM(CASE WHEN tsh IS NULL THEN 1 ELSE 0 END) as null_tsh,
    SUM(CASE WHEN t3 IS NULL THEN 1 ELSE 0 END) as null_t3,
    SUM(CASE WHEN tt4 IS NULL THEN 1 ELSE 0 END) as null_tt4,
    SUM(CASE WHEN t4u IS NULL THEN 1 ELSE 0 END) as null_t4u,
    SUM(CASE WHEN fti IS NULL THEN 1 ELSE 0 END) as null_fti
FROM thyroid_numeric
UNION ALL
SELECT 
    'thyroid_categorical' as table_name,
    COUNT(*) as total_rows,
    SUM(CASE WHEN sex IS NULL THEN 1 ELSE 0 END) as null_sex,
    0 as null_tsh, 0 as null_t3, 0 as null_tt4, 0 as null_t4u, 0 as null_fti
FROM thyroid_categorical
UNION ALL
SELECT 
    'thyroid_labels' as table_name,
    COUNT(*) as total_rows,
    SUM(CASE WHEN class IS NULL THEN 1 ELSE 0 END) as null_class,
    0 as null_tsh, 0 as null_t3, 0 as null_tt4, 0 as null_t4u, 0 as null_fti
FROM thyroid_labels;