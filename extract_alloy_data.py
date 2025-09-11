#!/usr/bin/env python3
"""
Script to extract alloy-wise data from WAAM.json
Organizes data by alloy type and extracts specified parameters
"""

import json
import re
import sys
from collections import defaultdict

def extract_numeric_value(value_str):
    """Extract numeric values from strings that may contain multiple values or ranges"""
    if not value_str or value_str == "":
        return ""
    
    # Convert to string if it's a number
    value_str = str(value_str)
    
    # Handle ranges (e.g., "18.5-19.6")
    if '-' in value_str and not value_str.startswith('-'):
        parts = value_str.split('-')
        if len(parts) == 2:
            try:
                val1 = float(parts[0].strip())
                val2 = float(parts[1].strip())
                return (val1 + val2) / 2  # Return average
            except ValueError:
                pass
    
    # Handle multiple values separated by semicolons or commas
    if ';' in value_str:
        values = value_str.split(';')
        numeric_values = []
        for val in values:
            val = val.strip()
            # Extract first number from each part
            numbers = re.findall(r'-?\d+\.?\d*', val)
            if numbers:
                try:
                    numeric_values.append(float(numbers[0]))
                except ValueError:
                    pass
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)  # Return average
    
    # Handle comma-separated values
    if ',' in value_str:
        values = value_str.split(',')
        numeric_values = []
        for val in values:
            val = val.strip()
            # Extract first number from each part
            numbers = re.findall(r'-?\d+\.?\d*', val)
            if numbers:
                try:
                    numeric_values.append(float(numbers[0]))
                except ValueError:
                    pass
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)  # Return average
    
    # Handle single values with text (e.g., "Peak: 150, Background: 70, Average: 110")
    numbers = re.findall(r'-?\d+\.?\d*', value_str)
    if numbers:
        try:
            # If multiple numbers found, take the first one
            return float(numbers[0])
        except ValueError:
            pass
    
    # Try direct conversion
    try:
        return float(value_str)
    except ValueError:
        return value_str  # Return original if can't convert

def classify_alloy(material, composition):
    """Classify alloy based on material name and composition"""
    material = str(material).lower()
    composition = str(composition).lower()
    
    # Tin alloys (Sn-based)
    if any(keyword in material for keyword in ['sn', 'tin', 'pb', 'lead']):
        return "Tin Alloys"
    if any(keyword in composition for keyword in ['sn', 'tin', 'pb', 'lead']):
        return "Tin Alloys"
    
    # Steel alloys (any material containing steel, iron, or Fe-based)
    if any(keyword in material for keyword in ['steel', 'iron', 'fe-', 'mild steel', 'carbon steel', 'stainless']):
        return "Steel Alloys"
    if any(keyword in composition for keyword in ['fe', 'iron', 'steel']):
        return "Steel Alloys"
    
    # Titanium alloys
    if any(keyword in material for keyword in ['ti-', 'titanium']):
        return "Titanium Alloys"
    if 'ti' in composition and ('al' in composition or 'v' in composition):
        return "Titanium Alloys"
    
    # Aluminum alloys
    if any(keyword in material for keyword in ['al', 'aluminum', 'aluminium']):
        return "Aluminum Alloys"
    if 'al' in composition and not 'ti' in composition:
        return "Aluminum Alloys"
    
    # Nickel alloys
    if any(keyword in material for keyword in ['ni-', 'nickel', 'inconel']):
        return "Nickel Alloys"
    if 'ni' in composition and 'ni' not in 'titanium':
        return "Nickel Alloys"
    
    # Copper alloys
    if any(keyword in material for keyword in ['cu-', 'copper', 'brass', 'bronze']):
        return "Copper Alloys"
    if 'cu' in composition and 'cu' not in 'titanium':
        return "Copper Alloys"
    
    # Magnesium alloys
    if any(keyword in material for keyword in ['mg-', 'magnesium']):
        return "Magnesium Alloys"
    if 'mg' in composition:
        return "Magnesium Alloys"
    
    # Intermetallic alloys
    if any(keyword in material for keyword in ['intermetallic', 'fe-al', 'ti-al']):
        return "Intermetallic Alloys"
    
    # Default classification
    return "Other Alloys"

def extract_strength_data(strength_data):
    """Extract UTS and Elongation from strength data"""
    uts_waam = ""
    uts_bm = ""
    elong_waam = ""
    elong_bm = ""
    
    if isinstance(strength_data, dict):
        waam_deposition = strength_data.get("At WAAM Deposition", "")
        base_material = strength_data.get("At Base Material", "")
        
        # Convert to string if not already
        waam_deposition = str(waam_deposition) if waam_deposition else ""
        base_material = str(base_material) if base_material else ""
        
        # Extract UTS from WAAM deposition
        if waam_deposition:
            uts_match = re.search(r'UTS[:\s]*([0-9.-]+)', waam_deposition, re.IGNORECASE)
            if uts_match:
                uts_waam = extract_numeric_value(uts_match.group(1))
        
        # Extract UTS from base material
        if base_material:
            uts_match = re.search(r'UTS[:\s]*([0-9.-]+)', base_material, re.IGNORECASE)
            if uts_match:
                uts_bm = extract_numeric_value(uts_match.group(1))
        
        # Extract Elongation from WAAM deposition
        if waam_deposition:
            elong_match = re.search(r'elongation[:\s]*([0-9.-]+)', waam_deposition, re.IGNORECASE)
            if elong_match:
                elong_waam = extract_numeric_value(elong_match.group(1))
        
        # Extract Elongation from base material
        if base_material:
            elong_match = re.search(r'elongation[:\s]*([0-9.-]+)', base_material, re.IGNORECASE)
            if elong_match:
                elong_bm = extract_numeric_value(elong_match.group(1))
    
    return uts_waam, uts_bm, elong_waam, elong_bm

def main():
    # Read the WAAM.json file
    try:
        with open('/Users/azim/Desktop/MTP/WAAM.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded WAAM.json with {len(data)} entries")
        
    except FileNotFoundError:
        print("Error: WAAM.json file not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing WAAM.json: {e}")
        sys.exit(1)
    
    # Organize data by alloy type
    alloy_data = defaultdict(list)
    
    for entry in data:
        serial_no = entry.get("Serial No.", "")
        data_list = entry.get("Data", [])
        
        for data_item in data_list:
            # Get material information
            material_info = data_item.get("WAAM wise Material", {})
            material = material_info.get("Material", "")
            composition = material_info.get("Composition", "")
            
            # Classify alloy
            alloy_type = classify_alloy(material, composition)
            
            # Get welding parameters
            welding_params = data_item.get("Welding Parameters", {})
            heat_input = welding_params.get("Heat Input (kJ/mm)", "")
            power_kw = welding_params.get("Power(kW)", "")
            travel_speed = welding_params.get("Travel Speed (mm/s)", "")
            wire_diameter = welding_params.get("Wire Diameter (mm)", "")
            
            # Get strength data
            strength_data = data_item.get("Strength", {})
            uts_waam, uts_bm, elong_waam, elong_bm = extract_strength_data(strength_data)
            
            # Get bead dimensions
            bead_height = data_item.get("Bead Height", "")
            bead_width = data_item.get("Bead Width", "")
            overlap = data_item.get("Overlap (%)", "")
            
            # Create entry for this data item
            alloy_entry = {
                "Serial No.": serial_no,
                "Heat Input (kJ/mm)": extract_numeric_value(heat_input),
                "Power(kW)": extract_numeric_value(power_kw),
                "Travel Speed (mm/s)": extract_numeric_value(travel_speed),
                "Wire Diameter (mm)": extract_numeric_value(wire_diameter),
                "UTS(WAAM)(MPa)": uts_waam,
                "UTS(BM)(MPa)": uts_bm,
                "Elong(WAAM)(%)": elong_waam,
                "Elong(BM)(%)": elong_bm,
                "Bead Width(mm)": extract_numeric_value(bead_width),
                "Bead Height(mm)": extract_numeric_value(bead_height),
                "Overlap(%)": extract_numeric_value(overlap)
            }
            
            alloy_data[alloy_type].append(alloy_entry)
    
    # Create final output structure
    output_data = {}
    for alloy_type, entries in alloy_data.items():
        output_data[alloy_type] = entries
    
    # Write to output file
    output_file = '/Users/azim/Desktop/MTP/WAAM_alloy_data.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully extracted alloy-wise data")
        print(f"Output saved to: {output_file}")
        
        # Print summary statistics
        print(f"\nSummary by Alloy Type:")
        for alloy_type, entries in output_data.items():
            print(f"  {alloy_type}: {len(entries)} entries")
        
        # Count entries with data
        total_entries = sum(len(entries) for entries in output_data.values())
        entries_with_heat_input = sum(1 for entries in output_data.values() for entry in entries if entry["Heat Input (kJ/mm)"] != "")
        entries_with_power = sum(1 for entries in output_data.values() for entry in entries if entry["Power(kW)"] != "")
        entries_with_travel_speed = sum(1 for entries in output_data.values() for entry in entries if entry["Travel Speed (mm/s)"] != "")
        entries_with_uts_waam = sum(1 for entries in output_data.values() for entry in entries if entry["UTS(WAAM)(MPa)"] != "")
        
        print(f"\nData Availability:")
        print(f"  Total entries: {total_entries}")
        print(f"  Entries with Heat Input: {entries_with_heat_input}")
        print(f"  Entries with Power: {entries_with_power}")
        print(f"  Entries with Travel Speed: {entries_with_travel_speed}")
        print(f"  Entries with UTS(WAAM): {entries_with_uts_waam}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
