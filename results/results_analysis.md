# Evaluation results analysis

> Populate this report after `evaluation_results.csv` contains measured results.
> Do not substitute the paper-reported scores for our measurements.

## Results summary

- Highest Joint: pending measured results.
- Best STA: pending measured results.
- Best SIM: pending measured results.
- Best ChrF1: pending measured results.
- Generation failures: pending measured results.

## Comparison with COLING 2025

Paper-reported English Joint values (reference only):
- Human references: 0.711
- Duplicate: 0.061
- Delete: 0.447
- Backtranslation: 0.506
- condBERT: 0.278
- mBART-Translated: 0.443
- mBART-mParaDetox: 0.339
- GPT-4 few-shot: 0.475
- GPT-4 CoT: 0.326

Our `mbart_400` and `mbart_large_data` runs are not automatically equivalent to the
paper's mBART baselines. Compare them as separate experiments unless training
data, preprocessing, checkpoint, and decoding configuration match.

## Training data scale

Compare `mbart_400` and `mbart_large_data` on the same 600 IDs. Report changes in
STA, SIM, ChrF1, Joint, and failed generations. Do not claim that more data caused
an improvement from only one run per setting; seed and training differences may
also matter.

## Qualitative examples

Add a few examples selected from per-example metrics:
- Successful detoxification: high STA, reasonable SIM, and reasonable ChrF1.
- Over-deletion / meaning loss: potentially high STA but low SIM or ChrF1.
- Residual toxicity: low STA.
- Fluent paraphrase with low reference overlap: inspect manually before judging.

## Limitations

- Toxicity classifiers can misclassify context, reclaimed language, or subtle abuse.
- LaBSE similarity is not a guarantee of exact meaning preservation.
- ChrF measures reference overlap and may penalize valid paraphrases.
- Joint compresses several dimensions into one score and can hide trade-offs.
- A single 600-example test split provides limited uncertainty estimates.
- Report missing/failed generations; do not silently drop them.
