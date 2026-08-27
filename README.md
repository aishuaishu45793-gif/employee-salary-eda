# 💼 Employee Salary EDA & HR Analytics Dashboard

An end-to-end **Employee Salary Data Analysis and HR Analytics project** built using Python, Pandas, NumPy, Matplotlib, Seaborn, Pytest, and Streamlit.

The project performs data validation, data cleaning, feature engineering, exploratory data analysis, visualization, automated testing, and provides an interactive Streamlit dashboard for exploring employee salary and workforce patterns.

---

## 📌 Project Overview

Employee salary can be influenced by several workforce-related factors such as:

- Experience
- Age
- Projects completed
- Training hours
- Experience-related features

This project analyzes these relationships using a complete data analytics pipeline.

The project also includes an interactive **Streamlit dashboard** that allows users to explore employee salary patterns using filters and multiple analytical views.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Validate the quality of the raw employee dataset
- Identify missing values
- Detect duplicate records
- Detect negative and invalid values
- Clean and prepare the dataset
- Handle missing values
- Remove invalid records
- Remove duplicate records
- Create meaningful engineered features
- Analyze salary relationships
- Create statistical and visual insights
- Build automated tests using Pytest
- Create an interactive Streamlit dashboard
- Prepare model-ready datasets

---

# 🔄 Project Data Pipeline

```text
Raw Employee Dataset
        ↓
Data Validation
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Visualization
        ↓
Automated Testing
        ↓
Streamlit Dashboard
📊 Dataset

The project uses employee-level data containing the following fields:

Feature	Description
experience_years	Employee work experience in years
age	Employee age
projects_completed	Number of completed projects
training_hours	Employee training hours
salary_lakh	Employee salary in lakh
Engineered Features
Feature	Description
experience_score	Normalized experience score
project_experience_ratio	Relationship between projects completed and experience
🧹 Data Validation

The raw dataset was validated before cleaning.

The validation process checks:

Missing values
Duplicate rows
Negative values
Zero values
Invalid age values
Invalid experience values
Invalid project values
Invalid training values
Invalid salary values
Suspicious records
Initial Dataset
Rows    : 1020
Columns : 5
Validation Findings
Missing cells       : 42
Duplicate rows      : 20
Negative values     : 10

The validation script identifies problematic records before they enter the cleaning pipeline.

🧽 Data Cleaning

The cleaning pipeline removes invalid records and duplicate rows and then performs missing-value imputation.

Cleaning Results
Initial rows                 : 1020
After invalid-record removal : 1005
After duplicate removal     : 985
Missing cells after cleaning: 0
Duplicate rows after cleaning: 0
Invalid age                 : 0
Negative experience         : 0
Negative salary             : 0
Final Clean Dataset
985 rows × 5 columns

The cleaned dataset is saved as:

data/processed/employee_salary_cleaned.csv
⚙️ Feature Engineering

Two additional features were created.

1. Experience Score

experience_score represents normalized employee experience.

The resulting feature is validated to remain between:

0 and 1
2. Project Experience Ratio

project_experience_ratio represents the relationship between completed projects and employee experience.

The feature engineering pipeline validates that the ratio is:

Finite
Non-negative
Free from missing values
📦 Model-Ready Dataset

The final model-ready dataset contains:

X rows : 985
X columns : 6
y rows : 985
Input Features
experience_years
age
projects_completed
training_hours
experience_score
project_experience_ratio
Target
salary_lakh

The project also validates:

X/y row alignment
Missing values
Target leakage
Engineered feature ranges
📈 Exploratory Data Analysis

The project analyzes relationships between employee characteristics and salary.

The analysis includes:

Salary distribution
Experience distribution
Experience vs salary
Projects vs salary
Training vs salary
Correlation analysis
Experience-group salary analysis
Feature-to-salary relationships
🔥 Key Insights

The current analysis produced the following results:

Metric	Result
Employees	985
Average Salary	13.53 Lakh
Median Salary	13.60 Lakh
Average Experience	10.02 years
Experience-Salary Correlation	0.899
Projects-Salary Correlation	0.212
Training-Salary Correlation	0.100
Main Finding

Experience has the strongest observed relationship with salary in this dataset.

The experience-salary correlation is approximately:

0.899

Projects completed and training hours show weaker relationships with salary:

Projects → Salary : 0.212
Training → Salary : 0.100

Note: Correlation indicates association, not causation.

📊 Visualizations

The project generates the following visualizations:

reports/figures/
│
├── salary_distribution.png
├── experience_distribution.png
├── experience_vs_salary.png
├── projects_vs_salary.png
├── training_vs_salary.png
└── correlation_heatmap.png
🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Dashboard Modules

The dashboard provides:

Executive Overview
Employee Analytics
Compensation Analytics
Experience & Career Analytics
Projects & Productivity
Learning & Training
Advanced Analytics
Employee Explorer
HR Metrics Roadmap
Dashboard Features
Interactive employee filters
Experience filtering
Salary filtering
Age filtering
Project filtering
Training-hour filtering
KPI cards
Salary charts
Experience analysis
Project analysis
Training analysis
Correlation heatmap
Salary outlier detection
Employee data table
Filtered data download
HR analytics roadmap
📸 Dashboard Screenshots
Executive Overview

Compensation Analytics

Employee Analytics

Advanced Analytics

Employee Explorer

Add the screenshots to the screenshots/ folder using the filenames above.

🎥 Project Demo

A 2–3 minute demonstration video shows:

Data validation
Data cleaning
Feature engineering
Automated testing
Data analysis
Visualization
Streamlit dashboard
Interactive filtering
Employee analytics
▶️ Demo Video

Add your video link here:

YOUR_VIDEO_LINK
🧪 Automated Testing

Automated testing is implemented using Pytest.

The project currently contains:

11 tests

All tests passed successfully:

11 passed

The tests cover areas including:

Data cleaning
Invalid value handling
Missing-value handling
Duplicate handling
Feature engineering
Experience score
Project/experience ratio
Model-ready data
Feature validation
Target leakage
Data consistency

Run the tests with:

pytest

Expected result:

11 passed
📁 Project Structure
employee-salary-eda/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── 03_employee_salary.csv
│   │   └── employee_salary.csv
│   │
│   └── processed/
│       ├── employee_salary_cleaned.csv
│       └── employee_salary_model_ready.csv
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_cleaning.ipynb
│   └── 04_feature_engineering.ipynb
│
├── outputs/
│   ├── employee_salary_model_ready.csv
│   ├── x.csv
│   └── y.csv
│
├── reports/
│   └── figures/
│       ├── correlation_heatmap.png
│       ├── experience_distribution.png
│       ├── experience_vs_salary.png
│       ├── projects_vs_salary.png
│       ├── salary_distribution.png
│       └── training_vs_salary.png
│
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── cleaning.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   └── visualization.py
│
├── tests/
│   ├── test_cleaning.py
│   └── test_features.py
│
├── main.py
├── requirements.txt
└── README.md
🛠️ Technologies Used
Technology	Purpose
Python	Core programming language
Pandas	Data manipulation
NumPy	Numerical operations
Matplotlib	Data visualization
Seaborn	Statistical visualization
Streamlit	Interactive dashboard
Pytest	Automated testing
Jupyter Notebook	Exploratory analysis
Git	Version control
GitHub	Project repository
🚀 Installation
1. Clone the repository
git clone https://github.com/aishuaishu45793-gif/employee-salary-eda.git
2. Open the project
cd employee-salary-eda
3. Create a virtual environment
Windows
python -m venv .venv
4. Activate the environment
.venv\Scripts\activate
5. Install dependencies
pip install -r requirements.txt
▶️ Running the Project
Step 1 — Data Validation
python src/data_validation.py
Step 2 — Data Cleaning
python src/cleaning.py
Step 3 — Feature Engineering
python src/feature_engineering.py
Step 4 — Analysis
python src/analysis.py
Step 5 — Generate Visualizations
python src/visualization.py
Step 6 — Run Tests
pytest
Step 7 — Launch Dashboard
streamlit run app/app.py

The dashboard will normally be available at:

http://localhost:8501
🏢 HR Analytics Extension

The current dataset supports salary and workforce analysis.

A real company HR analytics system could be extended by integrating additional HR data sources.

Possible future metrics include:

Employee attrition
Employee turnover
Retention rate
Attendance rate
Absenteeism
Performance rating
Employee engagement
Recruitment analytics
Time to hire
Time to fill
Offer acceptance rate
Cost per hire
Promotion rate
Internal mobility
Training completion
Training ROI
Payroll cost
Benefits cost
Revenue per employee
Pay equity

These metrics require additional company HR data and are not calculated from the current dataset.

🔮 Future Improvements

Potential future improvements include:

Salary prediction using machine learning
Employee salary benchmarking
Employee segmentation
Model comparison
Feature importance analysis
Automated reporting
Additional HR datasets
Employee attrition prediction
Performance analytics
Recruitment analytics
Deployment of the Streamlit application
Database integration
Real-time HR analytics
📋 Project Status
Data Validation       ✅
Data Cleaning         ✅
Feature Engineering   ✅
EDA                   ✅
Visualization         ✅
Automated Testing     ✅
Streamlit Dashboard   ✅
GitHub Repository     ✅
👩‍💻 Author

Aishwarya

GitHub:

https://github.com/aishuaishu45793-gif
