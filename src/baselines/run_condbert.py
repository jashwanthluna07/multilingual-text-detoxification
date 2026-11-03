
import os
import pickle
import numpy as np
import torch
import pandas as pd
# Compatibility fix for legacy keras_preprocessing with NumPy 2.x
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from transformers import BertTokenizer, BertForMaskedLM
from condbert import CondBertRewriter
from multiword.masked_token_predictor_bert import MaskedTokenPredictorBert

# Paths
ROOT = r"D:\Downloads\assignment1_dataset_preparation"
VOCAB = "vocab"

INPUT = os.path.join(ROOT, "outputs", "duplicate_predictions.csv")

# CPU configuration
device = torch.device("cpu")

print("Loading BERT...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.to(device)
model.eval()

print("Loading vocabulary resources...")

with open(os.path.join(VOCAB, "negative-words.txt"), encoding="utf-8") as f:
    negative_words = [x.strip() for x in f if x.strip()]

with open(os.path.join(VOCAB, "toxic_words.txt"), encoding="utf-8") as f:
    negative_words += [x.strip() for x in f if x.strip()]

with open(os.path.join(VOCAB, "positive-words.txt"), encoding="utf-8") as f:
    positive_words = [x.strip() for x in f if x.strip()]

with open(os.path.join(VOCAB, "word2coef.pkl"), "rb") as f:
    word2coef = pickle.load(f)

token_toxicities = []

with open(os.path.join(VOCAB, "token_toxicities.txt")) as f:
    for line in f:
        token_toxicities.append(float(line.strip()))

token_toxicities = np.array(token_toxicities)
token_toxicities = np.maximum(
    0, np.log(1 / (1 / token_toxicities - 1))
)

for tok in [".", ",", "-"]:
    token_toxicities[tokenizer.encode(tok)][1] = 3

for tok in ["you"]:
    token_toxicities[tokenizer.encode(tok)][1] = 0

print("Initializing CondBERT...")

editor = CondBertRewriter(
    model=model,
    tokenizer=tokenizer,
    device=device,
    neg_words=negative_words,
    pos_words=positive_words,
    word2coef=word2coef,
    token_toxicities=token_toxicities
)

predictor = MaskedTokenPredictorBert(
    model,
    tokenizer,
    max_len=250,
    device=device,
    label=0,
    contrast_penalty=0.0
)

editor.predictor = predictor

def adjust_logits(logits, label=0):
    return logits - editor.token_toxicities * 3

predictor.logits_postprocessor = adjust_logits

# Read the already validated test predictions file.
df = pd.read_csv(INPUT)

print("Testing one sentence...")

sample = "You are an idiot!"

with torch.no_grad():
    result = editor.replacement_loop(
        sample,
        verbose=False,
        n_top=10,
        n_tokens=(1, 2, 3),
        n_units=1
    )

print("Original:", sample)
print("Detoxified:", result)

print("Single-sentence CondBERT test successful.")
# Full CondBERT evaluation on 600 test examples
test_path = r"D:\Downloads\assignment1_dataset_preparation\en-00000-of-00001 (1).parquet"
output_path = r"D:\Downloads\assignment1_dataset_preparation\outputs\condbert_predictions.csv"

test_df = pd.read_parquet(test_path)
results = []

for i, row in test_df.iterrows():
    toxic = str(row["toxic_sentence"])

    try:
        generated = editor.replacement_loop(
            toxic,
            verbose=False,
            n_top=10,
            n_tokens=(1, 2, 3),
            n_units=1
        )
    except Exception as e:
        print(f"Error on example {i + 1}: {e}")
        generated = toxic

    results.append({
        "id": i + 1,
        "toxic_sentence": toxic,
        "reference_neutral_sentence": row["neutral_sentence"],
        "generated_sentence": generated,
        "model_name": "CondBERT"
    })

    if (i + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Completed {i + 1}/{len(test_df)}")

pd.DataFrame(results).to_csv(output_path, index=False)
print("All CondBERT predictions saved:", output_path)