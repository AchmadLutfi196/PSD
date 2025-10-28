-- Quick Analysis: Thyroid Categorical Dataset Insights
-- Key findings from the categorical features

-- =====================================================
-- DATASET OVERVIEW
-- =====================================================

-- Total: 7,200 patients with 15 categorical features
-- All features are binary (0/1) representing absence/presence

SELECT 'DATASET OVERVIEW' as analysis_type;

-- =====================================================
-- KEY DEMOGRAPHIC INSIGHTS
-- =====================================================

-- Gender Distribution:
-- Female (0): 5,009 patients (69.6%)
-- Male (1): 2,191 patients (30.4%)
-- Observation: Female patients are significantly more represented

-- =====================================================
-- TREATMENT PREVALENCE ANALYSIS
-- =====================================================

-- Treatment Types (from most to least common):
-- 1. On Thyroxine: 940 patients (13.1%)
-- 2. I131 Treatment: 121 patients (1.7%)
-- 3. Query on Thyroxine: 111 patients (1.5%)
-- 4. Thyroid Surgery: 101 patients (1.4%)
-- 5. Anti-thyroid Medication: 92 patients (1.3%)

-- =====================================================
-- SYMPTOM/CONDITION PREVALENCE
-- =====================================================

-- Most Common Conditions:
-- 1. Query Hyperthyroid: 495 patients (6.9%)
-- 2. Query Hypothyroid: 472 patients (6.6%)
-- 3. Psych conditions: 352 patients (4.9%)
-- 4. Sick: 276 patients (3.8%)
-- 5. Tumor: 184 patients (2.6%)

-- Rare Conditions:
-- 1. Hypopituitary: 1 patient (0.01%) - Extremely rare
-- 2. Goitre: 59 patients (0.8%)
-- 3. Pregnant: 78 patients (1.1%)
-- 4. Lithium use: 91 patients (1.3%)

-- =====================================================
-- MEDICAL INSIGHTS
-- =====================================================

-- Key Observations:
-- 1. Thyroid hormone replacement (thyroxine) is the most common treatment
-- 2. Hyperthyroid queries are slightly more common than hypothyroid queries
-- 3. Psychological conditions affect nearly 5% of patients
-- 4. Pregnancy rate is low (1.1%), suggesting either male-dominated or post-reproductive age dataset
-- 5. Hypopituitary condition is extremely rare (only 1 case)

-- =====================================================
-- CLINICAL DECISION PATTERNS
-- =====================================================

-- Treatment Decision Tree Insights:
-- - 13.1% of patients are on thyroxine (hormone replacement therapy)
-- - Only 1.3% are on anti-thyroid medications (hyperthyroid treatment)
-- - This suggests the dataset may have more hypothyroid cases than hyperthyroid

-- Query Patterns:
-- - 6.9% have hyperthyroid queries vs 6.6% hypothyroid queries
-- - Nearly balanced, suggesting good diagnostic screening

-- =====================================================
-- FEATURE IMPORTANCE FOR ANALYSIS
-- =====================================================

-- High-Variance Features (most informative):
-- 1. on_thyroxine (13.1% positive rate)
-- 2. query_hyperthyroid (6.9%)
-- 3. query_hypothyroid (6.6%)
-- 4. psych (4.9%)
-- 5. sick (3.8%)

-- Low-Variance Features (less discriminative):
-- 1. hypopituitary (0.01%)
-- 2. goitre (0.8%)
-- 3. pregnant (1.1%)

-- =====================================================
-- POSTGRESQL QUERY EXAMPLES FOR ANALYSIS
-- =====================================================

/*
-- Example 1: Find patients with multiple risk factors
SELECT * FROM thyroid_categorical 
WHERE (sick + on_thyroxine + query_hypothyroid + query_hyperthyroid + psych) >= 2;

-- Example 2: Gender-based treatment analysis
SELECT 
    CASE WHEN sex = 0 THEN 'Female' ELSE 'Male' END as gender,
    COUNT(*) as total_patients,
    SUM(on_thyroxine) as on_thyroxine_count,
    ROUND(AVG(on_thyroxine) * 100, 1) as thyroxine_percentage
FROM thyroid_categorical 
GROUP BY sex;

-- Example 3: Treatment combination patterns
SELECT 
    on_thyroxine,
    on_antithyroid_medication,
    thyroid_surgery,
    i131_treatment,
    COUNT(*) as patient_count
FROM thyroid_categorical
WHERE on_thyroxine = 1 OR on_antithyroid_medication = 1 OR thyroid_surgery = 1 OR i131_treatment = 1
GROUP BY on_thyroxine, on_antithyroid_medication, thyroid_surgery, i131_treatment
ORDER BY patient_count DESC;

-- Example 4: Risk scoring system
SELECT 
    id,
    CASE WHEN sex = 0 THEN 'F' ELSE 'M' END as gender,
    (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment) as treatment_score,
    (query_hypothyroid + query_hyperthyroid + sick + psych + tumor + goitre) as symptom_score,
    CASE 
        WHEN (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment + 
              query_hypothyroid + query_hyperthyroid + sick + psych + tumor + goitre) >= 3 
        THEN 'High Risk'
        WHEN (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment + 
              query_hypothyroid + query_hyperthyroid + sick + psych + tumor + goitre) >= 1 
        THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_category
FROM thyroid_categorical
ORDER BY 
    (on_thyroxine + on_antithyroid_medication + thyroid_surgery + i131_treatment + 
     query_hypothyroid + query_hyperthyroid + sick + psych + tumor + goitre) DESC;
*/