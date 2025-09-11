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
    
    # Convert Heat Input and Bead Height to numeric, handling empty strings and non-numeric values
    df['Heat_Input_numeric'] = pd.to_numeric(df['Heat Input (kJ/mm)'], errors='coerce')
    df['Bead_Height_numeric'] = pd.to_numeric(df['Bead Height(mm)'], errors='coerce')
    
    # Remove rows where either Heat Input or Bead Height is NaN
    df_clean = df.dropna(subset=['Heat_Input_numeric', 'Bead_Height_numeric'])
    
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
    
    # Only remove the most extreme outliers that would completely distort the graph
    # Remove only extremely high heat input values (>2000 kJ/mm) and extreme bead heights (>50 mm or <0.01 mm)
    df_filtered = df[
        (df['Heat_Input_numeric'] <= 2000) & 
        (df['Bead_Height_numeric'] <= 50) &
        (df['Bead_Height_numeric'] >= 0.01)
    ]
    print(f"After manual filtering (Heat Input <= 2000 kJ/mm, Bead Height 0.01-50 mm): {df_filtered.shape[0]} points")
    
    # Apply very conservative statistical outlier removal only to extreme cases
    heat_outliers_iqr, h_lower, h_upper = detect_outliers_iqr(df_filtered, 'Heat_Input_numeric')
    heat_outliers_zscore = detect_outliers_zscore(df_filtered, 'Heat_Input_numeric', threshold=4.0)
    
    print(f"Heat Input outliers (IQR method): {len(heat_outliers_iqr)}")
    print(f"Heat Input outliers (Z-score method): {len(heat_outliers_zscore)}")
    
    # Only remove the most extreme outliers, keep as much data as possible
    df_no_outliers = df_filtered[
        (df_filtered['Heat_Input_numeric'] >= h_lower) & (df_filtered['Heat_Input_numeric'] <= h_upper)
    ]
    
    print("Data shape after outlier removal:", df_no_outliers.shape)
    
    # Print some statistics
    print("\nHeat Input statistics:")
    print(f"Min: {df_no_outliers['Heat_Input_numeric'].min():.2f} kJ/mm")
    print(f"Max: {df_no_outliers['Heat_Input_numeric'].max():.2f} kJ/mm")
    print(f"Mean: {df_no_outliers['Heat_Input_numeric'].mean():.2f} kJ/mm")
    
    print("\nBead Height statistics:")
    print(f"Min: {df_no_outliers['Bead_Height_numeric'].min():.2f} mm")
    print(f"Max: {df_no_outliers['Bead_Height_numeric'].max():.2f} mm")
    print(f"Mean: {df_no_outliers['Bead_Height_numeric'].mean():.2f} mm")
    
    return df_no_outliers

def create_heat_input_vs_bead_height_plot(df):
    """Create the Heat Input vs Bead Height plot with different colors and shapes for each alloy type"""
    
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
                alloy_data['Bead_Height_numeric'], 
                alloy_data['Heat_Input_numeric'],
                c=style['color'],
                marker=style['marker'],
                label=f'{alloy_type} (n={len(alloy_data)})',
                alpha=0.8,
                s=style['size'],
                edgecolors='black',
                linewidth=1.0
            )
    
    # Customize the plot
    plt.xlabel('Bead Height (mm)', fontsize=12, fontweight='bold')
    plt.ylabel('Heat Input (kJ/mm)', fontsize=12, fontweight='bold')
    plt.title('Heat Input vs Bead Height for Different WAAM Alloy Types\n(Distinct Colors and Shapes)', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('heat_input_vs_bead_height_plot.png', dpi=300, bbox_inches='tight')
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
        print(f"  Heat Input range: {alloy_data['Heat_Input_numeric'].min():.2f} - {alloy_data['Heat_Input_numeric'].max():.2f} kJ/mm")
        print(f"  Bead Height range: {alloy_data['Bead_Height_numeric'].min():.2f} - {alloy_data['Bead_Height_numeric'].max():.2f} mm")
        print(f"  Mean Heat Input: {alloy_data['Heat_Input_numeric'].mean():.2f} kJ/mm")
        print(f"  Mean Bead Height: {alloy_data['Bead_Height_numeric'].mean():.2f} mm")

def main():
    """Main function to execute the analysis and plotting"""
    print("Loading and cleaning WAAM alloy data for Heat Input vs Bead Height...")
    df = load_and_clean_data()
    
    print(f"\nLoaded {len(df)} data points with valid Heat Input and Bead Height values")
    
    print("\nHandling outliers...")
    df_clean = handle_outliers(df)
    
    print_data_summary(df_clean)
    
    print("\nCreating Heat Input vs Bead Height plot...")
    plt = create_heat_input_vs_bead_height_plot(df_clean)
    
    print("\nPlot saved as 'heat_input_vs_bead_height_plot.png'")
    print("Python script saved as 'heat_input_vs_bead_height_plot.py'")

if __name__ == "__main__":
    main()
