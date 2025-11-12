
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("Cleaned_Water_2109.xlsx", index_col=0)
#df = pd.read_excel("Merged_Water_Weather_Final.xlsx", index_col=0)

row_idx = np.arange(len(df))  # unique integer positions
train_pos, test_pos = train_test_split(
    row_idx, test_size=0.3, random_state=42
)

df_train = df.iloc[train_pos].copy()
df_test  = df.iloc[test_pos].copy()

df_train.to_excel("df_train.xlsx", index=True)
df_test.to_excel("df_test.xlsx", index=True)

print("\n Split Summary")
print(f"Train: {len(df_train):,}  |  Test: {len(df_test):,}  |  Total: {len(df):,}")
print("Overlap (row positions):", len(set(train_pos) & set(test_pos)))  # should be 0

