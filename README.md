**Business Performance Analysis**

This project analyzes business financial health by examining debt ratios and profitability metrics. The goal is to identify high-risk companies, clean and structure raw data for analysis, and generate insights at both the company and state level.

**Key Steps**

Imported and validated raw business financial data.

Cleaned and normalized data (handling missing values, duplicates, and anomalies).

Calculated Debt-to-Income (DTI) and Debt-to-Equity (D/E) ratios.

Flagged businesses with negative equity (negative D/E) as high-risk.

Aggregated results to generate state-level descriptive statistics.

Visualized relationships between profitability and leverage.

**Outputs**

negative_debt_to_equity_businesses.csv → list of high-risk businesses with negative equity.

business_level_with_DTI.csv → cleaned dataset with DTI calculated for each business.

state_descriptive_stats.csv → aggregated financial health indicators at the state level.

**Code Versions**

Business_Health_Analysis.py — initial version.

Business_Health_Analysis_v2.py — adds a division-by-zero safeguard for DTI and refactors code for clarity/readability.

**How to Run**

Install required packages
Make sure you have Python 3.10+ installed, then install the necessary libraries:

```bash
pip install pandas matplotlib numpy
Ensure dataset is available
Place the dataset file Business_Financials_Data.csv in the same folder as the notebook or script.

Run as a Jupyter Notebook
If you prefer to run the analysis interactively:

bash
Copy code
jupyter notebook business_performance_analysis_notebook.ipynb

Run as a Python script
If you want to execute the full pipeline as a script:

bash
Copy code
python Business_Health_Analysis_v2.py

**Tools & Skills Demonstrated**

Python (pandas, matplotlib) for data cleaning, transformation, and visualization.

Jupyter Notebook for reproducible analysis.

Financial analysis concepts (leverage ratios, profitability metrics).

Data storytelling — combining ratios and risk flags to interpret business health.

Developed using Python (pandas) in Jupyter Notebook This project is presented as part of my professional portfolio and is not a distributed solution.
