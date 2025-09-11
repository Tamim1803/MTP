#!/usr/bin/env python3
"""
Script to extract authors' institutes with their serial numbers from WAAM.json
"""

import json
import os

def extract_authors_institutes():
    """Extract authors' institutes with serial numbers from WAAM.json"""
    
    # Read the WAAM.json file
    with open('/Users/azim/Desktop/MTP/WAAM.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract authors' institutes with serial numbers
    authors_institutes = []
    
    for entry in data:
        serial_no = entry.get("Serial No.", "")
        
        # Check if there's data array
        if "Data" in entry and isinstance(entry["Data"], list):
            for data_entry in entry["Data"]:
                authors_institute = data_entry.get("Authors Institute", "")
                if authors_institute:  # Only add if institute is not empty
                    authors_institutes.append({
                        "Serial No.": serial_no,
                        "Authors Institute": authors_institute
                    })
        else:
            # Handle case where data might be directly in the entry
            authors_institute = entry.get("Authors Institute", "")
            if authors_institute:
                authors_institutes.append({
                    "Serial No.": serial_no,
                    "Authors Institute": authors_institute
                })
    
    return authors_institutes

def main():
    """Main function to extract and save the data"""
    try:
        # Extract the data
        authors_institutes = extract_authors_institutes()
        
        # Save to a new JSON file
        output_file = '/Users/azim/Desktop/MTP/authors_institutes.json'
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(authors_institutes, file, indent=2, ensure_ascii=False)
        
        print(f"Successfully extracted {len(authors_institutes)} entries")
        print(f"Data saved to: {output_file}")
        
        # Display first few entries as preview
        print("\nFirst 5 entries:")
        for i, entry in enumerate(authors_institutes[:5]):
            print(f"{i+1}. Serial No.: {entry['Serial No.']}")
            print(f"   Authors Institute: {entry['Authors Institute']}")
            print()
            
    except FileNotFoundError:
        print("Error: WAAM.json file not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in WAAM.json")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

