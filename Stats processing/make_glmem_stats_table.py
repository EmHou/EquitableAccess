# equitability/table.py
# Extract and display GLMEM statistical results in table format

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path to import analysis functions
sys.path.append(str(Path(__file__).parent.parent))

def extract_glmem_table_results():
    """
    Extract GLMEM results and format them into a table with Estimate, SE, 95% CI (LL, UL), and p-values.
    """
    
    # Load the GLMEM results file
    results_dir = Path(__file__).parent
    glmem_results_path = results_dir / "multimodal_glmem_socioeconomic_health_equity_results.txt"
    
    if not glmem_results_path.exists():
        print(f"ERROR: GLMEM results file not found at: {glmem_results_path}")
        print("   Please run the GLMEM analysis first to generate results.")
        return None
    
    print(f" Loading GLMEM results from: {glmem_results_path}")
    
    # Read the GLMEM results file
    with open(glmem_results_path, 'r') as f:
        content = f.read()
    
    print(f" Loaded GLMEM results file: {len(content)} characters")
    
    # Parse the content to extract results
    results_data = []
    
    # Split content into sections by mode
    sections = content.split("MODE:")
    
    for section in sections:
        if not section.strip():
            continue
            
        # Extract mode name
        lines = section.split("\n")
        if len(lines) < 2:
            continue
            
        mode = lines[0].strip()
        if not mode:
            continue
            
        # Look for socioeconomic indicators and SVI themes in this section
        current_section = ""
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a socioeconomic indicator
            if line.startswith("Socioeconomic_") and ":" in line:
                current_section = "socioeconomic"
                indicator_name = line.split(":")[0].replace("Socioeconomic_", "")
                continue
                
            # Check if this is an SVI theme
            elif line.startswith("SVI_") and ":" in line:
                current_section = "svi"
                theme_name = line.split(":")[0].replace("SVI_", "")
                continue
                
            # Extract coefficient
            elif line.startswith("Coefficient:"):
                coef = float(line.split(":")[1].strip())
                continue
                
            # Extract standard error
            elif line.startswith("Standard Error:"):
                se = float(line.split(":")[1].strip())
                continue
                
            # Extract 95% CI
            elif line.startswith("95% CI:"):
                ci_part = line.split(":")[1].strip()
                ci_ll = float(ci_part.split(",")[0].replace("(", "").strip())
                ci_ul = float(ci_part.split(",")[1].replace(")", "").strip())
                continue
                
            # Extract p-value
            elif line.startswith("P-value:"):
                pval = float(line.split(":")[1].strip())
                
                # Now we have all the data, add to results
                if current_section == "socioeconomic":
                    results_data.append({
                        'Transportation_Mode': mode,
                        'Variable_Type': 'Socioeconomic',
                        'Variable_Name': indicator_name,
                        'Estimate': coef,
                        'SE': se,
                        'CI_LL': ci_ll,
                        'CI_UL': ci_ul,
                        'P_value': pval
                    })
                elif current_section == "svi":
                    results_data.append({
                        'Transportation_Mode': mode,
                        'Variable_Type': 'SVI_Theme',
                        'Variable_Name': theme_name,
                        'Estimate': coef,
                        'SE': se,
                        'CI_LL': ci_ll,
                        'CI_UL': ci_ul,
                        'P_value': pval
                    })
                
                # Reset for next variable
                current_section = ""
                indicator_name = ""
                theme_name = ""
                coef = se = ci_ll = ci_ul = pval = None
    
    if not results_data:
        print("ERROR: No GLMEM results found in the file")
        return None
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    print(f" Extracted {len(results_df)} GLMEM results")
    print(f"   Transportation modes: {results_df['Transportation_Mode'].unique()}")
    print(f"   Variable types: {results_df['Variable_Type'].unique()}")
    
    return results_df

def create_formatted_table(results_df):
    """
    Create a formatted table with Estimate, SE, 95% CI (LL, UL), and p-values.
    """
    if results_df is None or len(results_df) == 0:
        return None
    
    # Create a copy for formatting
    table_df = results_df.copy()
    
    # Format p-values
    def format_p_value(pval):
        if pd.isna(pval):
            return "N/A"
        elif pval < 0.001:
            return "<0.001"
        elif pval < 0.01:
            return f"{pval:.3f}"
        elif pval < 0.05:
            return f"{pval:.3f}"
        else:
            return f"{pval:.3f}"
    
    # Format estimates and other numeric values
    def format_numeric(val):
        if pd.isna(val):
            return "N/A"
        else:
            return f"{val:.3f}"
    
    # Apply formatting
    table_df['Estimate_Formatted'] = table_df['Estimate'].apply(format_numeric)
    table_df['SE_Formatted'] = table_df['SE'].apply(format_numeric)
    table_df['CI_Formatted'] = table_df.apply(
        lambda row: f"({format_numeric(row['CI_LL'])}, {format_numeric(row['CI_UL'])})" 
        if not pd.isna(row['CI_LL']) and not pd.isna(row['CI_UL']) else "N/A", axis=1
    )
    table_df['P_value_Formatted'] = table_df['P_value'].apply(format_p_value)
    
    # Create the final table
    final_table = table_df[[
        'Transportation_Mode', 'Variable_Type', 'Variable_Name',
        'Estimate_Formatted', 'SE_Formatted', 'CI_Formatted', 'P_value_Formatted'
    ]].copy()
    
    # Rename columns for display
    final_table.columns = [
        'Transportation Mode', 'Variable Type', 'Variable Name',
        'Estimate', 'SE', '95% CI (LL, UL)', 'P-value'
    ]
    
    return final_table

def save_table_to_csv(table_df, filename="glmem_statistical_table.csv"):
    """
    Save the formatted table to CSV.
    """
    if table_df is None:
        return
    
    results_dir = Path(__file__).parent
    
    filepath = results_dir / filename
    table_df.to_csv(filepath, index=False)
    print(f" Table saved to: {filepath}")
    
    return filepath

def display_table(table_df):
    """
    Display the formatted table in the console.
    """
    if table_df is None:
        return
    
    print("\n" + "=" * 100)
    print("GLMEM STATISTICAL RESULTS TABLE")
    print("=" * 100)
    
    # Display the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(table_df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print(" Complete statistical results extracted including Estimate, SE, 95% CI, and P-values")

def main():
    """
    Main function to run the table extraction and display.
    """
    print(" GLMEM Statistical Table Generator")
    print("=" * 50)
    
    # Extract results
    results_df = extract_glmem_table_results()
    
    if results_df is None:
        print("\n To generate the table:")
        print("   1. First run the GLMEM analysis (equitability/analysis.py)")
        print("   2. Then run this table generator")
        return
    
    # Create formatted table
    table_df = create_formatted_table(results_df)
    
    if table_df is None:
        return
    
    # Display table
    display_table(table_df)
    
    # Save to CSV
    save_table_to_csv(table_df)
    
    print("\n Table generation complete!")
    print("\n Complete statistical table generated with Estimate, SE, 95% CI (LL, UL), and P-values")
    print("   This matches the format used in other research papers")

if __name__ == "__main__":
    main()
