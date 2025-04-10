# Re-import libraries after code environment reset
import pandas as pd

# Reload the Excel file to inspect sheet names
elb_path = "WISAData_ELB.xlsx"
elb_xls = pd.ExcelFile(elb_path)
elb_xls.sheet_names

# Load the 'Daten Ebene 1 und Stammdaten' sheet
elb_df = pd.read_excel(elb_path, sheet_name='Daten Ebene 1 und Stammdaten')

# Display structure and quick preview
elb_df.info(), elb_df.head()

# Clean the ELB dataset:
# - Promote the fourth row (index 3) as header
# - Drop top formatting rows
# - Reset index

elb_df_cleaned = elb_df[4:].copy()
elb_df_cleaned.columns = elb_df.iloc[3]
elb_df_cleaned.reset_index(drop=True, inplace=True)

# Clean column names
elb_df_cleaned.columns = [str(col).strip() for col in elb_df_cleaned.columns]

# Extract chemical data (from column 10 onwards)
elb_chem_data = elb_df_cleaned[elb_df_cleaned.columns[10:]]
elb_chem_data = elb_chem_data.apply(pd.to_numeric, errors='coerce')

# Analyze missing values
missing_counts_elb = elb_chem_data.isnull().sum()
missing_percent_elb = (missing_counts_elb / len(elb_chem_data)) * 100

missing_summary_elb = pd.DataFrame({
    'Missing Count': missing_counts_elb,
    'Missing %': missing_percent_elb
}).sort_values(by='Missing Count', ascending=False)

# Filter well-filled parameters
well_filled_columns_elb = missing_summary_elb[missing_summary_elb["Missing %"] < 20].index
elb_well_filled_data = elb_chem_data[well_filled_columns_elb]

# Generate descriptive statistics
elb_descriptive_stats = elb_well_filled_data.describe().transpose()

# Save results
excel_output_elb = "Descriptive_Statistics_ELB.xlsx"
elb_descriptive_stats.to_excel(excel_output_elb)

