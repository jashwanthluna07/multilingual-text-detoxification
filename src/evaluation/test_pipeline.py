import pandas as pd

test = pd.read_csv("data/test.csv")

predictions = pd.read_csv("predictions/predictions_sample.csv")

print("Test examples:", len(test))
print("Prediction examples:", len(predictions))

# Check IDs and columns
assert "id" in test.columns
assert "toxic_sentence" in test.columns
assert "neutral_sentence" in test.columns

assert "id" in predictions.columns
assert "generated_sentence" in predictions.columns

# Match predictions to the test examples by ID
merged = test.merge(
    predictions,
    on="id",
    how="left",
    validate="one_to_one"
)

merged["generation_failed"] = (
    merged["generated_sentence"].isna()
    | merged["generated_sentence"].astype(str).str.strip().eq("")
)

print("\nMatched examples:", len(merged))
print("Failed generations:", merged["generation_failed"].sum())

print("\nSample joined results:")
print(
    merged[
        ["id", "toxic_sentence", "neutral_sentence",
         "generated_sentence", "generation_failed"]
    ].head().to_string(index=False)
)

assert len(merged) == 600

print("\nDataset and ID-alignment checks passed!")