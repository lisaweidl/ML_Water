import pandas as pd
from openpyxl import load_workbook

# Load Excel file
excel_path = "WISAData_MAR.xlsx"
xls = pd.ExcelFile(excel_path)
df = pd.read_excel(xls, sheet_name="Daten Ebene 1 und Stammdaten")

# Promote second row to header and drop top formatting rows
df_cleaned = df[2:].copy()
df_cleaned.columns = df.iloc[1]
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.columns = [str(col).strip() for col in df_cleaned.columns]

# Only chemical measurement columns
chem_data = df_cleaned[df_cleaned.columns[10:]]
chem_data = chem_data.apply(pd.to_numeric, errors='coerce')

# Handle missing values
missing_counts = chem_data.isnull().sum()
missing_percent = (missing_counts / len(chem_data)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent
}).sort_values(by='Missing Count', ascending=False)

# Filter parameters with at least 80% completeness
well_filled_columns = missing_summary[missing_summary["Missing %"] < 20].index
well_filled_data = chem_data[well_filled_columns]
descriptive_stats = well_filled_data.describe().transpose()

# Save to Excel
output_path = "Descriptive_Statistics_MAR.xlsx"
descriptive_stats.to_excel(output_path)

# Add description to the top of the Excel file
workbook = load_workbook(output_path)
sheet = workbook.active
sheet.insert_rows(1, amount=3)
sheet["A1"] = "This is the original case study area MAR."
sheet["A2"] = "It includes quantitative and qualitative groundwater data for 10 monitoring sites."
sheet["A3"] = "There are 20 different parameters measured. In total, there are 830 groundwater samples with over 1,400 datapoints."
workbook.save(output_path)
