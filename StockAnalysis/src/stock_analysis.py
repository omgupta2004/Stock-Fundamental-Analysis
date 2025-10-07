import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# File path
FILE_PATH = r"C:\Users\omgup\OneDrive\Desktop\StockAnalysis\data\data.xls"

def load_and_prepare_data():
    """Load and clean data from Excel sheets"""
    print("\nLOADING AND PREPARING DATA\n")
    
    # Load sheets
    sheet1 = pd.read_excel(FILE_PATH, sheet_name='Sheet1')
    sheet2 = pd.read_excel(FILE_PATH, sheet_name='Sheet2')
    
    # Clean stock prices
    stock_prices = sheet1.iloc[4:].copy()
    stock_prices.columns = ['Date', 'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM']
    stock_prices['Date'] = pd.to_datetime(stock_prices['Date'], format='%m/%d/%Y')
    stock_prices = stock_prices.set_index('Date')
    
    for col in stock_prices.columns:
        stock_prices[col] = pd.to_numeric(stock_prices[col], errors='coerce')
    
    # Prepare fundamentals
    def get_metric(raw, metric):
        df = raw[raw['Field'] == metric].copy()
        df['Ticker'] = df['Ticker'].str.upper()
        df = df.set_index('Ticker')
        years = [c for c in df.columns if isinstance(c, int)]
        return df[years].T
    
    sales_df = get_metric(sheet2, 'SALES')
    ebitda_df = get_metric(sheet2, 'EBITDA')
    pat_df = get_metric(sheet2, 'PAT')
    
    print(f"Stock prices loaded: {stock_prices.shape[0]} days")
    print(f"Fundamentals loaded: {sales_df.shape[0]} years")
    
    return stock_prices, sales_df, ebitda_df, pat_df

def calculate_fundamental_variables(sales_df, ebitda_df, pat_df):
    """Calculate growth rates and margin changes"""
    print("\nCALCULATING FUNDAMENTAL VARIABLES\n")
    
    # Growth rates (YoY percentage change)
    sales_growth = sales_df.pct_change(periods=-1) * 100
    ebitda_growth = ebitda_df.pct_change(periods=-1) * 100
    pat_growth = pat_df.pct_change(periods=-1) * 100
    
    # Margins
    ebitda_margin = (ebitda_df / sales_df) * 100
    pat_margin = (pat_df / sales_df) * 100
    
    # Margin changes (YoY)
    ebitda_margin_change = ebitda_margin.diff(periods=-1)
    pat_margin_change = pat_margin.diff(periods=-1)
    
    print("Variables calculated:")
    print("- Sales Growth")
    print("- EBITDA Growth")
    print("- PAT Growth")
    print("- EBITDA Margin Change")
    print("- PAT Margin Change")
    
    return sales_growth, ebitda_margin_change, ebitda_growth, pat_growth, pat_margin_change

def prepare_analysis_data(stock_prices, sales_growth, ebitda_margin_change, ebitda_growth, pat_growth, pat_margin_change):
    """Align stock prices with fundamental data"""
    # Resample to yearly
    stock_prices_yearly = stock_prices.resample('Y').mean()
    stock_prices_yearly.index = stock_prices_yearly.index.year
    
    # Find common years
    years = set(stock_prices_yearly.index)
    years = years.intersection(set(sales_growth.index))
    years = years.intersection(set(ebitda_growth.index))
    years = years.intersection(set(pat_growth.index))
    years = years.intersection(set(ebitda_margin_change.index))
    years = years.intersection(set(pat_margin_change.index))
    years = sorted(list(years))
    
    # Filter to common years
    stock_prices_yearly = stock_prices_yearly.loc[years]
    
    metrics = {
        'sales_growth': sales_growth.loc[years],
        'ebitda_margin_change': ebitda_margin_change.loc[years],
        'ebitda_growth': ebitda_growth.loc[years],
        'pat_growth': pat_growth.loc[years],
        'pat_margin_change': pat_margin_change.loc[years]
    }
    
    print(f"\nAnalysis period: {min(years)} to {max(years)} ({len(years)} years)")
    
    return stock_prices_yearly, metrics

def run_correlation_analysis(stock_prices_yearly, metrics):
    """Run correlation analysis for each company"""
    print("\nCORRELATION ANALYSIS\n")
    
    companies = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM']
    results = {}
    
    for co in companies:
        df = pd.DataFrame({
            'Stock_Price': stock_prices_yearly[co],
            'sales_growth': metrics['sales_growth'][co],
            'ebitda_margin_change': metrics['ebitda_margin_change'][co],
            'ebitda_growth': metrics['ebitda_growth'][co],
            'pat_growth': metrics['pat_growth'][co],
            'pat_margin_change': metrics['pat_margin_change'][co]
        }).dropna()
        
        corr_matrix = df.corr()
        results[co] = corr_matrix
        
        print(f"{co} - Correlations with Stock Price:")
        print(corr_matrix['Stock_Price'].drop('Stock_Price').round(4))
        print()
    
    return results

def visualize_correlations(corr_results):
    """Create correlation heatmaps"""
    companies = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, co in enumerate(companies):
        sns.heatmap(corr_results[co], annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=1, ax=axes[idx])
        axes[idx].set_title(f'{co} - Correlation Matrix', fontsize=14, fontweight='bold')
    
    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig('correlation_matrices.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmaps saved as 'correlation_matrices.png'\n")
    plt.show()

def run_regression_analysis(stock_prices_yearly, metrics):
    """Run linear regression for each company"""
    print("\nREGRESSION ANALYSIS\n")
    
    companies = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM']
    fundamental_vars = ['sales_growth', 'ebitda_margin_change', 'ebitda_growth', 'pat_growth', 'pat_margin_change']
    results = {}
    
    for co in companies:
        # Prepare data
        X = pd.DataFrame({v: metrics[v][co] for v in fundamental_vars})
        y = stock_prices_yearly[co]
        df = pd.concat([y, X], axis=1).dropna()
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run regression
        model = LinearRegression(fit_intercept=True)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        r2 = model.score(X_scaled, y)
        n = len(y)
        p = X_scaled.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        resids = y - y_pred
        rmse = np.sqrt(np.mean(resids ** 2))
        
        # Calculate t-statistics and p-values
        var_resids = np.sum(resids ** 2) / (n - p - 1)
        var_coef = var_resids * np.linalg.inv(X_scaled.T @ X_scaled).diagonal()
        se_coef = np.sqrt(var_coef)
        t_stats = model.coef_ / se_coef
        p_values = [2 * (1 - stats.t.cdf(abs(t), n - p - 1)) for t in t_stats]
        
        # Significance markers
        sig = ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' for p in p_values]
        
        # Create results dataframe
        coef_df = pd.DataFrame({
            'Variable': fundamental_vars,
            'Coefficient': model.coef_,
            'Std_Error': se_coef,
            't_statistic': t_stats,
            'p_value': p_values,
            'Significant': sig
        }).sort_values('p_value')
        
        results[co] = coef_df
        results[co + '_metrics'] = {
            'R2': r2,
            'Adj_R2': adj_r2,
            'RMSE': rmse,
            'Sample_Size': n,
            'Top_3_Variables': coef_df.head(3)['Variable'].tolist()
        }
        
        print(f"{co}:")
        print(f"  R² = {r2:.4f}")
        print(f"  Adjusted R² = {adj_r2:.4f}")
        print(f"  RMSE = {rmse:.2f}")
        print(f"  Sample Size = {n}")
        print(f"  Top 3 Significant Variables: {', '.join(coef_df.head(3)['Variable'].tolist())}")
        print()
    
    return results

def visualize_regression_results(reg_results):
    """Create regression visualizations"""
    companies = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM']
    
    # Plot 1: Coefficients comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, co in enumerate(companies):
        df = reg_results[co]
        colors = ['red' if p < 0.05 else 'gray' for p in df['p_value']]
        
        axes[idx].barh(df['Variable'], df['Coefficient'], color=colors)
        axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[idx].set_xlabel('Coefficient Value', fontsize=10)
        axes[idx].set_title(f'{co} - Regression Coefficients\n(Red = p<0.05)', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
    
    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig('regression_coefficients.png', dpi=300, bbox_inches='tight')
    print("Regression coefficients saved as 'regression_coefficients.png'\n")
    plt.show()
    
    # Plot 2: R-squared comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    r2_data = [reg_results[co + '_metrics']['R2'] for co in companies]
    adj_r2_data = [reg_results[co + '_metrics']['Adj_R2'] for co in companies]
    
    x = np.arange(len(companies))
    width = 0.35
    
    ax.bar(x - width/2, r2_data, width, label='R²', color='steelblue')
    ax.bar(x + width/2, adj_r2_data, width, label='Adjusted R²', color='coral')
    
    ax.set_xlabel('Company', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(companies)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('r_squared_comparison.png', dpi=300, bbox_inches='tight')
    print("R² comparison saved as 'r_squared_comparison.png'\n")
    plt.show()

def create_summary_tables(corr_results, reg_results):
    """Create and save summary tables"""
    print("\nCREATING SUMMARY TABLES\n")
    
    companies = ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM']
    
    # Correlation summary
    corr_summary = []
    for co in companies:
        corr_matrix = corr_results[co]
        stock_corrs = corr_matrix['Stock_Price'].drop('Stock_Price')
        for var, val in stock_corrs.items():
            corr_summary.append({'Company': co, 'Variable': var, 'Correlation': val})
    
    corr_df = pd.DataFrame(corr_summary)
    corr_pivot = corr_df.pivot(index='Variable', columns='Company', values='Correlation')
    corr_pivot.to_csv('correlation_summary.csv')
    print("Correlation Summary Table:")
    print(corr_pivot.round(4))
    print("\nSaved as 'correlation_summary.csv'\n")
    
    # Regression summary
    reg_summary = []
    for co in companies:
        metrics = reg_results[co + '_metrics']
        reg_summary.append({
            'Company': co,
            'R_Squared': metrics['R2'],
            'Adj_R_Squared': metrics['Adj_R2'],
            'RMSE': metrics['RMSE'],
            'Sample_Size': metrics['Sample_Size'],
            'Top_Var_1': metrics['Top_3_Variables'][0],
            'Top_Var_2': metrics['Top_3_Variables'][1],
            'Top_Var_3': metrics['Top_3_Variables'][2]
        })
    
    reg_df = pd.DataFrame(reg_summary)
    reg_df.to_csv('regression_summary.csv', index=False)
    print("Regression Summary Table:")
    print(reg_df.to_string(index=False))
    print("\nSaved as 'regression_summary.csv'\n")
    
    # Save detailed coefficients
    for co in companies:
        reg_results[co].to_csv(f'{co}_coefficients.csv', index=False)
    print("Detailed coefficient tables saved for each company\n")

def main():
    """Main execution function"""
    print("\nSTOCK PRICE vs FUNDAMENTALS ANALYSIS\n")
    print("Algorithms Used:")
    print("- Pearson Correlation Coefficient")
    print("- Ordinary Least Squares (OLS) Linear Regression")
    print("\nLibraries Used:")
    print("- pandas, numpy (data manipulation)")
    print("- scikit-learn (LinearRegression, StandardScaler)")
    print("- scipy (statistical tests)")
    print("- matplotlib, seaborn (visualization)")
    
    # Execute analysis pipeline
    stock_prices, sales_df, ebitda_df, pat_df = load_and_prepare_data()
    
    sales_growth, ebitda_margin_change, ebitda_growth, pat_growth, pat_margin_change = \
        calculate_fundamental_variables(sales_df, ebitda_df, pat_df)
    
    stock_prices_yearly, metrics = prepare_analysis_data(
        stock_prices, sales_growth, ebitda_margin_change, 
        ebitda_growth, pat_growth, pat_margin_change
    )
    
    corr_results = run_correlation_analysis(stock_prices_yearly, metrics)
    visualize_correlations(corr_results)
    
    reg_results = run_regression_analysis(stock_prices_yearly, metrics)
    visualize_regression_results(reg_results)
    
    create_summary_tables(corr_results, reg_results)
    
    print("\nANALYSIS COMPLETE!")
    print("\nOutput Files Generated:")
    print("- correlation_matrices.png")
    print("- regression_coefficients.png")
    print("- r_squared_comparison.png")
    print("- correlation_summary.csv")
    print("- regression_summary.csv")
    print("- [COMPANY]_coefficients.csv (for each company)")

if __name__ == "__main__":
    main()
