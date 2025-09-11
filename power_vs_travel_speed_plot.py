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
    
    # Convert Power and Travel Speed to numeric, handling empty strings and non-numeric values
    df['Power_numeric'] = pd.to_numeric(df['Power(kW)'], errors='coerce')
    df['Travel_Speed_numeric'] = pd.to_numeric(df['Travel Speed (mm/s)'], errors='coerce')
    
    # Remove rows where either Power or Travel Speed is NaN
    df_clean = df.dropna(subset=['Power_numeric', 'Travel_Speed_numeric'])
    
    return df_clean

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

def handle_outliers(df):
    """Handle outliers by removing extreme values that would distort the plot scale"""
    print("Original data shape:", df.shape)
    
    # Check for outliers in Power
    power_outliers_iqr, p_lower, p_upper = detect_outliers_iqr(df, 'Power_numeric')
    power_outliers_zscore = detect_outliers_zscore(df, 'Power_numeric', threshold=3)
    
    print(f"Power outliers (IQR method): {len(power_outliers_iqr)}")
    print(f"Power outliers (Z-score method): {len(power_outliers_zscore)}")
    
    # Check for outliers in Travel Speed
    speed_outliers_iqr, s_lower, s_upper = detect_outliers_iqr(df, 'Travel_Speed_numeric')
    speed_outliers_zscore = detect_outliers_zscore(df, 'Travel_Speed_numeric', threshold=3)
    
    print(f"Travel Speed outliers (IQR method): {len(speed_outliers_iqr)}")
    print(f"Travel Speed outliers (Z-score method): {len(speed_outliers_zscore)}")
    
    # Use IQR method for more conservative outlier removal
    # Remove outliers that are extreme in either Power or Travel Speed
    df_no_outliers = df[
        (df['Power_numeric'] >= p_lower) & (df['Power_numeric'] <= p_upper) &
        (df['Travel_Speed_numeric'] >= s_lower) & (df['Travel_Speed_numeric'] <= s_upper)
    ]
    
    print("Data shape after outlier removal:", df_no_outliers.shape)
    
    # Print some statistics
    print("\nPower statistics:")
    print(f"Min: {df_no_outliers['Power_numeric'].min():.2f} kW")
    print(f"Max: {df_no_outliers['Power_numeric'].max():.2f} kW")
    print(f"Mean: {df_no_outliers['Power_numeric'].mean():.2f} kW")
    
    print("\nTravel Speed statistics:")
    print(f"Min: {df_no_outliers['Travel_Speed_numeric'].min():.2f} mm/s")
    print(f"Max: {df_no_outliers['Travel_Speed_numeric'].max():.2f} mm/s")
    print(f"Mean: {df_no_outliers['Travel_Speed_numeric'].mean():.2f} mm/s")
    
    return df_no_outliers

def create_power_vs_travel_speed_plot(df):
    """Create the Power vs Travel Speed plot with different colors and shapes for each alloy type"""
    
    # Define colors and shapes for each alloy type
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
                alloy_data['Travel_Speed_numeric'], 
                alloy_data['Power_numeric'],
                c=style['color'],
                marker=style['marker'],
                label=f'{alloy_type} (n={len(alloy_data)})',
                alpha=0.8,
                s=style['size'],
                edgecolors='black',
                linewidth=1.0
            )
    
    # Customize the plot
    plt.xlabel('Travel Speed (mm/s)', fontsize=12, fontweight='bold')
    plt.ylabel('Power (kW)', fontsize=12, fontweight='bold')
    plt.title('Power vs Travel Speed for Different WAAM Alloy Types\n(Distinct Colors and Shapes)', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('power_vs_travel_speed_plot.png', dpi=300, bbox_inches='tight')
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
        print(f"  Power range: {alloy_data['Power_numeric'].min():.2f} - {alloy_data['Power_numeric'].max():.2f} kW")
        print(f"  Travel Speed range: {alloy_data['Travel_Speed_numeric'].min():.2f} - {alloy_data['Travel_Speed_numeric'].max():.2f} mm/s")
        print(f"  Mean Power: {alloy_data['Power_numeric'].mean():.2f} kW")
        print(f"  Mean Travel Speed: {alloy_data['Travel_Speed_numeric'].mean():.2f} mm/s")

def main():
    """Main function to execute the analysis and plotting"""
    print("Loading and cleaning WAAM alloy data...")
    df = load_and_clean_data()
    
    print(f"\nLoaded {len(df)} data points with valid Power and Travel Speed values")
    
    print("\nHandling outliers...")
    df_clean = handle_outliers(df)
    
    print_data_summary(df_clean)
    
    print("\nCreating Power vs Travel Speed plot...")
    plt = create_power_vs_travel_speed_plot(df_clean)
    
    print("\nPlot saved as 'power_vs_travel_speed_plot.png'")
    print("Python script saved as 'power_vs_travel_speed_plot.py'")

if __name__ == "__main__":
    main()
