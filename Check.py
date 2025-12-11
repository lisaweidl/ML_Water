import pandas as pd

def inspect_excel(df_Water):
    df = pd.read_excel(df_Water)

    print("="*80)
    print("SHAPE (Rows, Columns)")
    print(df.shape)

    print("\n" + "="*80)
    print("COLUMN TYPES")
    print(df.dtypes)

    print("\n" + "="*80)
    print("FIRST 5 ROWS")
    print(df.head())

    print("\n" + "="*80)
    print("MISSING VALUES PER COLUMN")
    print(df.isna().sum())

    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print(df.describe(include=['number','object', 'category']).T)

    print("="*80)
    print("DONE.")


inspect_excel("df_Water.xlsx")
