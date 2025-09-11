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
    
    # Convert UTS (WAAM) and UTS (BM) to numeric, handling empty strings and non-numeric values
    df['UTS_WAAM_numeric'] = pd.to_numeric(df['UTS(WAAM)(MPa)'], errors='coerce')
    df['UTS_BM_numeric'] = pd.to_numeric(df['UTS(BM)(MPa)'], errors='coerce')
    
    # Remove rows where either UTS (WAAM) or UTS (BM) is NaN
    df_clean = df.dropna(subset=['UTS_WAAM_numeric', 'UTS_BM_numeric'])
    
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
    # Remove very high UTS values (>2000 MPa) for both WAAM and BM
    df_filtered = df[
        (df['UTS_WAAM_numeric'] <= 2000) & 
        (df['UTS_BM_numeric'] <= 2000) &
        (df['UTS_WAAM_numeric'] >= 0) &
        (df['UTS_BM_numeric'] >= 0)
    ]
    print(f"After manual filtering (UTS <= 2000 MPa for both): {df_filtered.shape[0]} points")
    
    # Only apply statistical outlier removal to UTS (WAAM), not UTS (BM) to preserve variation
    uts_waam_outliers_iqr, w_lower, w_upper = detect_outliers_iqr(df_filtered, 'UTS_WAAM_numeric')
    uts_waam_outliers_zscore = detect_outliers_zscore(df_filtered, 'UTS_WAAM_numeric', threshold=3.0)
    
    print(f"UTS (WAAM) outliers (IQR method): {len(uts_waam_outliers_iqr)}")
    print(f"UTS (WAAM) outliers (Z-score method): {len(uts_waam_outliers_zscore)}")
    
    # Only remove UTS (WAAM) outliers, keep all UTS (BM) variations
    df_no_outliers = df_filtered[
        (df_filtered['UTS_WAAM_numeric'] >= w_lower) & (df_filtered['UTS_WAAM_numeric'] <= w_upper)
    ]
    
    print("Data shape after outlier removal:", df_no_outliers.shape)
    
    # Print some statistics
    print("\nUTS (WAAM) statistics:")
    print(f"Min: {df_no_outliers['UTS_WAAM_numeric'].min():.2f} MPa")
    print(f"Max: {df_no_outliers['UTS_WAAM_numeric'].max():.2f} MPa")
    print(f"Mean: {df_no_outliers['UTS_WAAM_numeric'].mean():.2f} MPa")
    
    print("\nUTS (BM) statistics:")
    print(f"Min: {df_no_outliers['UTS_BM_numeric'].min():.2f} MPa")
    print(f"Max: {df_no_outliers['UTS_BM_numeric'].max():.2f} MPa")
    print(f"Mean: {df_no_outliers['UTS_BM_numeric'].mean():.2f} MPa")
    
    return df_no_outliers

def create_uts_waam_vs_uts_bm_plot(df):
    """Create the UTS (WAAM) vs UTS (BM) plot with different colors and shapes for each alloy type"""
    
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
                alloy_data['UTS_BM_numeric'], 
                alloy_data['UTS_WAAM_numeric'],
                c=style['color'],
                marker=style['marker'],
                label=f'{alloy_type} (n={len(alloy_data)})',
                alpha=0.8,
                s=style['size'],
                edgecolors='black',
                linewidth=1.0
            )
    
    # Add a diagonal line (y=x) to show where WAAM = BM
    max_val = max(df['UTS_WAAM_numeric'].max(), df['UTS_BM_numeric'].max())
    min_val = min(df['UTS_WAAM_numeric'].min(), df['UTS_BM_numeric'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='WAAM = BM')
    
    # Customize the plot
    plt.xlabel('UTS (BM) (MPa)', fontsize=12, fontweight='bold')
    plt.ylabel('UTS (WAAM) (MPa)', fontsize=12, fontweight='bold')
    plt.title('UTS (WAAM) vs UTS (BM) for Different WAAM Alloy Types\n(Distinct Colors and Shapes)', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Make axes equal for better comparison
    plt.axis('equal')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('uts_waam_vs_uts_bm_plot.png', dpi=300, bbox_inches='tight')
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
        print(f"  UTS (WAAM) range: {alloy_data['UTS_WAAM_numeric'].min():.2f} - {alloy_data['UTS_WAAM_numeric'].max():.2f} MPa")
        print(f"  UTS (BM) range: {alloy_data['UTS_BM_numeric'].min():.2f} - {alloy_data['UTS_BM_numeric'].max():.2f} MPa")
        print(f"  Mean UTS (WAAM): {alloy_data['UTS_WAAM_numeric'].mean():.2f} MPa")
        print(f"  Mean UTS (BM): {alloy_data['UTS_BM_numeric'].mean():.2f} MPa")
        
        # Calculate strength ratio (WAAM/BM)
        if alloy_data['UTS_BM_numeric'].mean() > 0:
            strength_ratio = alloy_data['UTS_WAAM_numeric'].mean() / alloy_data['UTS_BM_numeric'].mean()
            print(f"  Strength ratio (WAAM/BM): {strength_ratio:.3f}")

def main():
    """Main function to execute the analysis and plotting"""
    print("Loading and cleaning WAAM alloy data for UTS (WAAM) vs UTS (BM)...")
    df = load_and_clean_data()
    
    print(f"\nLoaded {len(df)} data points with valid UTS (WAAM) and UTS (BM) values")
    
    print("\nHandling outliers...")
    df_clean = handle_outliers(df)
    
    print_data_summary(df_clean)
    
    print("\nCreating UTS (WAAM) vs UTS (BM) plot...")
    plt = create_uts_waam_vs_uts_bm_plot(df_clean)
    
    print("\nPlot saved as 'uts_waam_vs_uts_bm_plot.png'")
    print("Python script saved as 'uts_waam_vs_uts_bm_plot.py'")

if __name__ == "__main__":
    main()
