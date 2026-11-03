
from pathlib import Path
import pandas as pd

# Input and output paths
data_dir = Path("data")
input_file = data_dir / "en-00000-of-00001.parquet"
output_file = data_dir / "test.csv"

# Check that the dataset exists
if not input_file.exists():
    raise FileNotFoundError(
        f"Dataset not found: {input_file.resolve()}"
    )

# Load the English test dataset
df = pd.read_parquet(input_file)

# Verify the expected columns
required_columns = ["toxic_sentence", "neutral_sentence"]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {list(df.columns)}"
    )

# Check dataset size
print("Original dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

if len(df) != 600:
    raise ValueError(
        f"Expected 600 test examples, but found {len(df)}. "
        "Verify the dataset split before continuing."
    )

# Check for empty sentences
if df[required_columns].isna().any().any():
    raise ValueError("Dataset contains missing sentence values.")

if (df[required_columns].astype(str).apply(
    lambda col: col.str.strip().eq("")
).any().any()):
    raise ValueError("Dataset contains empty sentences.")

# Add stable IDs because the source file has no explicit ID column
df.insert(0, "id", [f"en_{i:03d}" for i in range(len(df))])

# Save as CSV
df.to_csv(output_file, index=False, encoding="utf-8")

# Verify the saved file
check = pd.read_csv(output_file)

print("\nConversion successful!")
print("Output:", output_file.resolve())
print("Rows:", len(check))
print("Columns:", check.columns.tolist())
print("\nFirst five examples:")
print(check.head().to_string(index=False))
