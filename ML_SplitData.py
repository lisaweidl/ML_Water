
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("df_Water.xlsx", index_col=0)

row_idx = np.arange(len(df))
train_pos, test_pos = train_test_split(
    row_idx, test_size=0.3, random_state=42
)

df_train = df.iloc[train_pos].copy()
df_test  = df.iloc[test_pos].copy()

# ONLY FOR JOINED DATA (ELMA)
#train_means = df_train.mean(numeric_only=True)
#df_train = df_train.fillna(train_means)
#df_test  = df_test.fillna(train_means)
#df_train = df_train.fillna(0)
#df_test  = df_test.fillna(0)

df_train.to_excel("df_Water_train.xlsx", index=True)
df_test.to_excel("df_Water_test.xlsx", index=True)

print("\nSplit Summary")
print(f"Train: {len(df_train):,}  |  Test: {len(df_test):,}  |  Total: {len(df):,}")
print("Overlap (row positions):", len(set(train_pos) & set(test_pos)))

