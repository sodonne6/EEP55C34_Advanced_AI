import pandas as pd

val_path = r"C:\Users\irish\Computer_Electronic_Engineering_Year5\semester_2\Advanced_AI\project\.hf\hub\datasets--PSewmuthu--How2Sign_Holistic\snapshots\96a0da665eba6e5bc9bd4bf6803546e74b222bbf\how2sign_holistic_features\metadata\how2sign_realigned_val.csv"

df = pd.read_csv(val_path, sep="\t")

print("rows:", len(df))
print("columns:", list(df.columns))
print("\nhead:")
print(df.head(3).to_string(index=False))
