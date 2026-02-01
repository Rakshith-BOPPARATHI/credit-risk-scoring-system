-- ================================================================
-- CREDIT RISK SCORING SYSTEM - MySQL Implementation
-- ================================================================
DROP DATABASE IF EXISTS credit_risk_db;
CREATE DATABASE credit_risk_db;
USE credit_risk_db;

-- ================================================================
-- 1. APPLICANTS TABLE
-- ================================================================
CREATE TABLE applicants (
    applicant_id INT AUTO_INCREMENT PRIMARY KEY,
    income DECIMAL(12, 2) NOT NULL,
    loan_amount DECIMAL(12, 2) NOT NULL,
    credit_history DECIMAL(5, 2) NOT NULL,
    work_experience ENUM('0-2 years', '2-5 years', '5+ years') NOT NULL,
    home_ownership ENUM('Rent', 'Mortgage', 'Own') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- 2. CREDIT SCORES TABLE
-- ================================================================
CREATE TABLE credit_scores (
    score_id INT AUTO_INCREMENT PRIMARY KEY,
    applicant_id INT NOT NULL,
    probability_of_default DECIMAL(12, 6) NOT NULL,
    credit_score INT NOT NULL,
    risk_category ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id)
);

-- ================================================================
-- 3. MODEL COEFFICIENTS TABLE
-- ================================================================
CREATE TABLE model_coefficients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    feature_name VARCHAR(50) NOT NULL,
    coefficient DECIMAL(12, 6) NOT NULL,
    mean_value DECIMAL(12, 6) NOT NULL,
    std_value DECIMAL(12, 6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert trained model coefficients (from Colab training)
INSERT INTO model_coefficients (feature_name, coefficient, mean_value, std_value) VALUES
('Intercept', 2.751131, 0.0, 1.0),
('Income', -1.112291, 60263.938441, 19827.051862),
('LoanAmount', 1.712167, 152740.615647, 50259.021197),
('CreditHistory', -0.912897, 14.741371, 8.774844),
('WorkExperience', 0.094141, 1.041250, 0.819786),
('HomeOwnership', 0.248981, 1.007500, 0.815441);

-- ================================================================
-- 4. SIGMOID FUNCTION (Core of Logistic Regression)
-- ================================================================
DELIMITER $$
CREATE FUNCTION sigmoid(z DECIMAL(12, 6))
RETURNS DECIMAL(12, 6)
DETERMINISTIC
BEGIN
    DECLARE result DECIMAL(12, 6);
    SET result = 1.0 / (1.0 + EXP(-z));
    RETURN result;
END$$
DELIMITER ;

-- ================================================================
-- 5. CALCULATE CREDIT SCORE PROCEDURE
-- ================================================================
DELIMITER $$
CREATE PROCEDURE calculate_credit_score(
    IN p_applicant_id INT,
    OUT p_probability DECIMAL(12, 6),
    OUT p_score INT,
    OUT p_risk VARCHAR(10)
)
BEGIN
    DECLARE v_income DECIMAL(12, 2);
    DECLARE v_loan_amount DECIMAL(12, 2);
    DECLARE v_credit_history DECIMAL(5, 2);
    DECLARE v_work_exp INT;
    DECLARE v_home_own INT;
    DECLARE v_income_scaled DECIMAL(12, 6);
    DECLARE v_loan_scaled DECIMAL(12, 6);
    DECLARE v_credit_scaled DECIMAL(12, 6);
    DECLARE v_work_scaled DECIMAL(12, 6);
    DECLARE v_home_scaled DECIMAL(12, 6);
    DECLARE v_z DECIMAL(12, 6);
    
    -- Get applicant data
    SELECT income, loan_amount, credit_history,
           CASE work_experience
               WHEN '0-2 years' THEN 0
               WHEN '2-5 years' THEN 1
               WHEN '5+ years' THEN 2
           END,
           CASE home_ownership
               WHEN 'Rent' THEN 0
               WHEN 'Mortgage' THEN 1
               WHEN 'Own' THEN 2
           END
    INTO v_income, v_loan_amount, v_credit_history, v_work_exp, v_home_own
    FROM applicants
    WHERE applicant_id = p_applicant_id;
    
    -- Scale features using stored parameters
    SET v_income_scaled = (v_income - 60263.938441) / 19827.051862;
    SET v_loan_scaled = (v_loan_amount - 152740.615647) / 50259.021197;
    SET v_credit_scaled = (v_credit_history - 14.741371) / 8.774844;
    SET v_work_scaled = (v_work_exp - 1.041250) / 0.819786;
    SET v_home_scaled = (v_home_own - 1.007500) / 0.815441;
    
    -- Calculate linear combination (z)
    SET v_z = 2.751131 +
              (-1.112291 * v_income_scaled) +
              (1.712167 * v_loan_scaled) +
              (-0.912897 * v_credit_scaled) +
              (0.094141 * v_work_scaled) +
              (0.248981 * v_home_scaled);
    
    -- Apply sigmoid function to get probability
    SET p_probability = sigmoid(v_z);
    
    -- Convert to credit score (0-100)
    SET p_score = ROUND(p_probability * 100);
    
    -- Determine risk category
    SET p_risk = CASE
        WHEN p_score < 30 THEN 'LOW'
        WHEN p_score >= 30 AND p_score < 70 THEN 'MEDIUM'
        ELSE 'HIGH'
    END;
    
    -- Store result
    INSERT INTO credit_scores (applicant_id, probability_of_default, credit_score, risk_category)
    VALUES (p_applicant_id, p_probability, p_score, p_risk);
END$$
DELIMITER ;

-- ================================================================
-- 6. TEST DATA - Sample Applicant
-- ================================================================
INSERT INTO applicants (income, loan_amount, credit_history, work_experience, home_ownership)
VALUES (75000.00, 150000.00, 10.5, '5+ years', 'Mortgage');

-- Test the scoring system
CALL calculate_credit_score(1, @prob, @score, @risk);
SELECT @prob AS probability_of_default, @score AS credit_score, @risk AS risk_category;

-- Query results
SELECT * FROM credit_scores;
