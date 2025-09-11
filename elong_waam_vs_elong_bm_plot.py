import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load the WAAM alloy data and clean it for plotting"""
    with open('WAAM_alloy_data.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier manipulation
    all_data = []
    for alloy_type, entries in data.items():
        for entry in entries:
            entry['Alloy_Type'] = alloy_type
            all_data.append(entry)
    
    df = pd.DataFrame(all_data)
    
    # Convert Elong(WAAM) and Elong(BM) to numeric, handling empty strings and non-numeric values
    df['Elong_WAAM_numeric'] = pd.to_numeric(df['Elong(WAAM)(%)'], errors='coerce')
    df['Elong_BM_numeric'] = pd.to_numeric(df['Elong(BM)(%)'], errors='coerce')
    
    # Remove rows where either Elong(WAAM) or Elong(BM) is NaN
    df_clean = df.dropna(subset=['Elong_WAAM_numeric', 'Elong_BM_numeric'])
    
    return df_clean

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    # Use 1.5 for moderate outlier removal
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3.0):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

def handle_outliers(df):
    """Handle outliers by removing extreme values that would distort the plot scale"""
    print("Original data shape:", df.shape)
    
    # First, remove only the most extreme outliers manually for better scaling
    # Remove very high elongation values (>50%) for both WAAM and BM
    df_filtered = df[
        (df['Elong_WAAM_numeric'] <= 50) & 
        (df['Elong_BM_numeric'] <= 50) &
        (df['Elong_WAAM_numeric'] >= 0) &
        (df['Elong_BM_numeric'] >= 0)
    ]
    print(f"After manual filtering (Elongation <= 50% for both): {df_filtered.shape[0]} points")
    
    # Only apply statistical outlier removal to Elong(WAAM), not Elong(BM) to preserve variation
    elong_waam_outliers_iqr, w_lower, w_upper = detect_outliers_iqr(df_filtered, 'Elong_WAAM_numeric')
    elong_waam_outliers_zscore = detect_outliers_zscore(df_filtered, 'Elong_WAAM_numeric', threshold=3.0)
    
    print(f"Elong(WAAM) outliers (IQR method): {len(elong_waam_outliers_iqr)}")
    print(f"Elong(WAAM) outliers (Z-score method): {len(elong_waam_outliers_zscore)}")
    
    # Only remove Elong(WAAM) outliers, keep all Elong(BM) variations
    df_no_outliers = df_filtered[
        (df_filtered['Elong_WAAM_numeric'] >= w_lower) & (df_filtered['Elong_WAAM_numeric'] <= w_upper)
    ]
    
    print("Data shape after outlier removal:", df_no_outliers.shape)
    
    # Print some statistics
    print("\nElong(WAAM) statistics:")
    print(f"Min: {df_no_outliers['Elong_WAAM_numeric'].min():.2f} %")
    print(f"Max: {df_no_outliers['Elong_WAAM_numeric'].max():.2f} %")
    print(f"Mean: {df_no_outliers['Elong_WAAM_numeric'].mean():.2f} %")
    
    print("\nElong(BM) statistics:")
    print(f"Min: {df_no_outliers['Elong_BM_numeric'].min():.2f} %")
    print(f"Max: {df_no_outliers['Elong_BM_numeric'].max():.2f} %")
    print(f"Mean: {df_no_outliers['Elong_BM_numeric'].mean():.2f} %")
    
    return df_no_outliers

def create_elong_waam_vs_elong_bm_plot(df):
    """Create the Elong(WAAM) vs Elong(BM) plot with different colors and shapes for each alloy type"""
    
    # Define colors and shapes for each alloy type (same as other plots)
    alloy_styles = {
        'Titanium Alloys': {'color': '#FF6B6B', 'marker': 'o', 'size': 80},      # Red circles
        'Steel Alloys': {'color': '#4ECDC4', 'marker': 's', 'size': 60},         # Teal squares
        'Aluminum Alloys': {'color': '#45B7D1', 'marker': '^', 'size': 80},      # Blue triangles
        'Other Alloys': {'color': '#96CEB4', 'marker': 'D', 'size': 80},         # Green diamonds
        'Tin Alloys': {'color': '#FFEAA7', 'marker': 'v', 'size': 80}            # Yellow inverted triangles
    }
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot each alloy type separately
    for alloy_type in df['Alloy_Type'].unique():
        alloy_data = df[df['Alloy_Type'] == alloy_type]
        
        if len(alloy_data) > 0:
            style = alloy_styles.get(alloy_type, {'color': '#000000', 'marker': 'o', 'size': 60})
            plt.scatter(
                alloy_data['Elong_BM_numeric'], 
                alloy_data['Elong_WAAM_numeric'],
                c=style['color'],
                marker=style['marker'],
                label=f'{alloy_type} (n={len(alloy_data)})',
                alpha=0.8,
                s=style['size'],
                edgecolors='black',
                linewidth=1.0
            )
    
    # Add a diagonal line (y=x) to show where WAAM = BM
    max_val = max(df['Elong_WAAM_numeric'].max(), df['Elong_BM_numeric'].max())
    min_val = min(df['Elong_WAAM_numeric'].min(), df['Elong_BM_numeric'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='WAAM = BM')
    
    # Customize the plot
    plt.xlabel('Elongation (BM) (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Elongation (WAAM) (%)', fontsize=12, fontweight='bold')
    plt.title('Elongation (WAAM) vs Elongation (BM) for Different WAAM Alloy Types\n(Distinct Colors and Shapes)', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Make axes equal for better comparison
    plt.axis('equal')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('elong_waam_vs_elong_bm_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return plt

def print_data_summary(df):
    """Print summary statistics for each alloy type"""
    print("\n" + "="*60)
    print("DATA SUMMARY BY ALLOY TYPE")
    print("="*60)
    
    for alloy_type in df['Alloy_Type'].unique():
        alloy_data = df[df['Alloy_Type'] == alloy_type]
        print(f"\n{alloy_type}:")
        print(f"  Number of data points: {len(alloy_data)}")
        print(f"  Elong(WAAM) range: {alloy_data['Elong_WAAM_numeric'].min():.2f} - {alloy_data['Elong_WAAM_numeric'].max():.2f} %")
        print(f"  Elong(BM) range: {alloy_data['Elong_BM_numeric'].min():.2f} - {alloy_data['Elong_BM_numeric'].max():.2f} %")
        print(f"  Mean Elong(WAAM): {alloy_data['Elong_WAAM_numeric'].mean():.2f} %")
        print(f"  Mean Elong(BM): {alloy_data['Elong_BM_numeric'].mean():.2f} %")
        
        # Calculate elongation ratio (WAAM/BM)
        if alloy_data['Elong_BM_numeric'].mean() > 0:
            elongation_ratio = alloy_data['Elong_WAAM_numeric'].mean() / alloy_data['Elong_BM_numeric'].mean()
            print(f"  Elongation ratio (WAAM/BM): {elongation_ratio:.3f}")

def main():
    """Main function to execute the analysis and plotting"""
    print("Loading and cleaning WAAM alloy data for Elongation (WAAM) vs Elongation (BM)...")
    df = load_and_clean_data()
    
    print(f"\nLoaded {len(df)} data points with valid Elongation (WAAM) and Elongation (BM) values")
    
    print("\nHandling outliers...")
    df_clean = handle_outliers(df)
    
    print_data_summary(df_clean)
    
    print("\nCreating Elongation (WAAM) vs Elongation (BM) plot...")
    plt = create_elong_waam_vs_elong_bm_plot(df_clean)
    
    print("\nPlot saved as 'elong_waam_vs_elong_bm_plot.png'")
    print("Python script saved as 'elong_waam_vs_elong_bm_plot.py'")

if __name__ == "__main__":
    main()
