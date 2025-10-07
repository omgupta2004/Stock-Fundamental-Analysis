
# Stock Price vs Fundamentals Analysis

> *A comprehensive statistical analysis examining the relationship between stock prices and fundamental financial metrics for Indian IT companies*

##  Project Overview

This project analyzes how financial fundamentals like sales growth, profit margins, and EBITDA impact stock prices of five leading Indian IT companies: **TCS**, **Infosys**, **HCL Technologies**, **Wipro**, and **Tech Mahindra**.

Using 20 years of data (2005-2024), I built correlation matrices and regression models to identify which fundamental metrics are the strongest predictors of stock performance for each company.

## Key Questions Answered

- **Which fundamental metrics correlate most strongly with stock prices?**
- **Can we predict stock movements using financial fundamentals?**
- **Do different companies respond differently to the same metrics?**
- **What are the top 3 most significant drivers for each company?**


##  Quick Start

### Prerequisites

- Python 3.8 or higher
- Basic understanding of statistics and finance

### Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/Stock-Fundamental-Analysis.git
cd Stock-Fundamental-Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python stock_analysis.py
```

The script will generate all output files in the current directory.

##  Methodology

### Data Collection

- **Stock Prices**: Daily closing prices from 2005-2025 (4,936 observations)
- **Fundamentals**: Annual financial data from 2005-2024 (29 years)
- **Companies**: TCS, Infosys, HCL Tech, Wipro, Tech Mahindra

### Variables Analyzed

I constructed five fundamental metrics:

1. **Sales Growth** - Year-over-year revenue change (%)
2. **EBITDA Growth** - Year-over-year EBITDA change (%)
3. **PAT Growth** - Year-over-year profit after tax change (%)
4. **EBITDA Margin Change** - Change in EBITDA margin (percentage points)
5. **PAT Margin Change** - Change in PAT margin (percentage points)

### Statistical Techniques

**Correlation Analysis**
- Used Pearson correlation to measure linear relationships
- Generated correlation matrices for each company
- Significance testing at α = 0.05 level

**Regression Analysis**
- Built multiple linear regression models (5 predictors per company)
- Applied StandardScaler for feature normalization
- Evaluated with R², Adjusted R², and RMSE
- Identified top 3 most significant variables using p-values

##  Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.12** | Primary programming language |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computations |
| **scikit-learn** | Machine learning (LinearRegression, StandardScaler) |
| **scipy** | Statistical tests and hypothesis testing |
| **matplotlib** | Data visualization |
| **seaborn** | Statistical graphics and heatmaps |

##  Results

### Output Files

The analysis generates:

**Visual Outputs:**
- Correlation heatmaps for all 5 companies
- Regression coefficient plots showing significance levels
- R² comparison charts across companies

**Data Tables:**
- Correlation summary (CSV format)
- Regression metrics with top 3 variables per company
- Detailed coefficient tables with p-values and t-statistics

### Sample Insights

Each company shows unique patterns:
- Different fundamental metrics drive different stocks
- Some companies are more sensitive to profitability, others to growth
- R² values indicate how well fundamentals explain stock movements

## Learning Outcomes

Through this project, I gained hands-on experience with:
- Financial data analysis and preprocessing
- Statistical hypothesis testing
- Multiple linear regression modeling
- Data visualization best practices
- Python data science ecosystem

##  Future Enhancements

This analysis can be extended with:

- **Advanced Models**: Ridge/Lasso regression, Random Forest
- **Time-Series Analysis**: Rolling correlations, lag effects
- **Deep Learning**: LSTM networks for stock prediction
- **Interactive Dashboard**: Streamlit app for dynamic exploration
- **Additional Metrics**: Include technical indicators, macro variables

##  References

- **Data Source**: Historical financial data from company reports
- **Statistical Methods**: Based on standard econometric practices
- **Python Libraries**: See `requirements.txt` for versions

##  Author

**OM GUPTA**
- Computer Science Student
- Passionate about Data Science and Finance



##  Code Example

Here's a quick peek at the core analysis:

```python
# Load and prepare data
stock_prices, sales_df, ebitda_df, pat_df = load_and_prepare_data()

# Calculate fundamental variables
sales_growth, ebitda_margin_change, ebitda_growth, pat_growth, pat_margin_change = \
    calculate_fundamental_variables(sales_df, ebitda_df, pat_df)

# Run correlation analysis
corr_results = run_correlation_analysis(stock_prices_yearly, metrics)

# Build regression models
reg_results = run_regression_analysis(stock_prices_yearly, metrics)

# Generate visualizations and tables
visualize_correlations(corr_results)
visualize_regression_results(reg_results)
create_summary_tables(corr_results, reg_results)
```

***


** If you found this project helpful, please consider giving it a star!**


This README is written in a friendly, accessible tone while maintaining professionalism. It clearly explains what the project does, how to use it, and what was learned—perfect for showcasing on GitHub and impressing potential employers or instructors!
