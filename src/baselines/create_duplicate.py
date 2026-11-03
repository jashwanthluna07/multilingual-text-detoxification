import pandas as pd
from pathlib import Path

test_path = Path("data/test.csv")
output_path = Path("predictions/duplicate.csv")

# Load the same 600-example test set
test = pd.read_csv(test_path)

# Validate the required columns
required = ["id", "toxic_sentence"]

if not all(col in test.columns for col in required):
    raise ValueError(f"Required columns: {required}")

# Duplicate baseline: return the original input unchanged
predictions = test[["id", "toxic_sentence"]].rename(
    columns={"toxic_sentence": "generated_sentence"}
)

# Save predictions
output_path.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(output_path, index=False, encoding="utf-8")

print("Duplicate baseline predictions created!")
print("Number of examples:", len(predictions))
print("Output:", output_path.resolve())
print("\nFirst five predictions:")
print(predictions.head().to_string(index=False))

assert len(predictions) == 600
assert predictions["id"].is_unique
assert predictions["generated_sentence"].notna().all()
