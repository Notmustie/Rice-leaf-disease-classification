import pandas as pd
from pathlib import Path
from src.inference.rejection import apply_rejection_to_df, add_correctness_columns, decision_buckets, summarize_by_rejection

inp = Path("outputs/predictions.csv")
df = pd.read_csv(inp)

df = apply_rejection_to_df(df, threshold=0.60)
df = add_correctness_columns(df)

print("\n=== Decision buckets (labeled only) ===")
print(decision_buckets(df))

print("\n=== Summary by rejection ===")
print(summarize_by_rejection(df))
