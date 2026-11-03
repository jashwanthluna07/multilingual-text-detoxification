# Multilingual Text Detoxification: Reproduction and Experimental Evaluation

An experimental study of **toxic text detoxification** using traditional NLP baselines and transformer-based models.

This project investigates how different approaches can transform toxic text into neutral language while preserving the original meaning. It includes baseline implementations, mBART experiments, an evaluation pipeline, comparisons with published COLING 2025 results, and quantitative analysis.

---

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Methods and Baselines](#methods-and-baselines)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Evaluation](#running-the-evaluation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Research Reference](#research-reference)

---

## Overview

Online platforms contain toxic language that can discourage participation and create hostile communication environments.

Text detoxification aims to rewrite toxic sentences into neutral, non-toxic alternatives while preserving their intended meaning.

This project studies multiple detoxification approaches and evaluates their performance using automatic metrics.

The work is inspired by the COLING 2025 research paper:

**Multilingual and Explainable Text Detoxification with Parallel Corpora.**

Rather than claiming a complete reproduction of every experiment in the paper, this repository documents selected reproduction-oriented experiments and additional model evaluations.

## Research Objectives

1. Study different approaches to automatic text detoxification.
2. Evaluate traditional and transformer-based baselines.
3. Compare performance across toxicity reduction, semantic preservation, and reference similarity.
4. Compare selected results with the published COLING 2025 baselines.
5. Investigate the effect of training-data scale through mBART experiments.
6. Identify the strengths and limitations of different detoxification approaches.

## Methods and Baselines

The experiments include the following approaches:

| Method | Description |
|---|---|
| Duplicate | Returns the original toxic input without modification. |
| Delete | Removes toxic content using a deletion-based approach. |
| Backtranslation | Uses translation and backtranslation to generate alternative wording. |
| CondBERT | Uses a conditional BERT-based approach for detoxification. |
| mBART (400 pairs) | mBART experiment using a small training dataset. |
| mBART (Large Dataset) | mBART experiment using a larger custom training dataset. |
| Pretrained mBART | Pretrained mBART-based experimental baseline. |

The Duplicate baseline is included as a reference point rather than a successful detoxification system.

## Dataset

The project uses English toxic-neutral sentence pairs and an evaluation set based on the COLING research dataset.

| Dataset component | Size |
|---|---:|
| Small training split | 400 pairs |
| English evaluation set | 600 examples |
| Larger custom English dataset | Approximately 19,000 pairs |

The same 600-example evaluation set is used to compare the seven systems reported in this repository.

Dataset availability and redistribution are subject to the original dataset's licensing terms. Refer to `data/README.md` for dataset access information.

## Evaluation Metrics

Four metrics are used to evaluate the generated detoxified sentences.

### 1. STA — Style Transfer Accuracy

Measures the probability that a generated sentence is classified as neutral by a toxicity classifier.

Higher values indicate stronger predicted toxicity reduction.

### 2. SIM — Semantic Similarity

Measures semantic similarity between the original toxic sentence and its generated version using LaBSE sentence embeddings and cosine similarity.

Higher values indicate greater semantic preservation.

### 3. ChrF1 — Character-level F-score

Measures character n-gram overlap between the generated sentence and the human-written neutral reference.

Higher values indicate greater character-level similarity to the reference.

### 4. Joint Score

Combines the three metrics at the individual-example level:

`Joint_i = STA_i × SIM_i × ChrF1_i`

The final Joint score is the mean of the per-example Joint scores:

`Joint = mean(Joint_i)`

This rewards systems that balance toxicity reduction, semantic preservation, and reference similarity.

**Note:** These are automatic evaluation metrics. They do not replace human judgments of fluency, meaning preservation, or detoxification quality.

## Experimental Results

All seven systems were evaluated on 600 examples.

| Model | STA ↑ | SIM ↑ | ChrF1 ↑ | Joint ↑ |
|---|---:|---:|---:|---:|
| Backtranslation | **0.8113** | 0.8806 | **0.7053** | **0.5228** |
| Delete | 0.7598 | **0.9448** | 0.6925 | 0.5052 |
| CondBERT | 0.7847 | 0.8775 | 0.6403 | 0.4540 |
| Pretrained mBART | 0.4961 | 0.9336 | 0.6946 | 0.3391 |
| mBART (Large Dataset) | 0.4850 | 0.9225 | 0.6815 | 0.3253 |
| mBART (400 Pairs) | 0.1353 | 0.9607 | 0.6412 | 0.0849 |
| Duplicate | 0.0909 | 1.0000 | 0.6703 | 0.0610 |

**Best Joint Score:** Backtranslation — 0.5228.

The detailed evaluation results and per-example metrics are available in the `results/` directory.

## Key Findings

### 1. Backtranslation achieved the strongest overall performance

Backtranslation obtained the highest Joint score (0.5228) and STA (0.8113) among the evaluated systems.

This indicates a comparatively strong balance between predicted toxicity reduction, semantic preservation, and similarity to the human neutral references.

### 2. Semantic similarity alone is insufficient

The Duplicate baseline achieved a SIM score close to 1.0 because its output is identical to the original input.

However, its STA score was only 0.0909.

This demonstrates why semantic similarity alone is insufficient for evaluating text detoxification.

### 3. More training data did not automatically improve the mBART results

The larger-data mBART experiment did not outperform the pretrained mBART experiment in Joint score.

This observation suggests that dataset size alone does not determine performance. Data quality, training configuration, model initialization, and optimization may also influence results.

### 4. The small-data mBART experiment struggled with toxicity reduction

The 400-pair mBART experiment achieved relatively high semantic similarity (0.9607), but a low STA score (0.1353).

This suggests that preserving the original sentence's meaning did not necessarily result in neutral outputs.

### 5. Reproduction results can differ from published results

The repository includes comparisons with selected English baselines reported in the original COLING 2025 paper.

Some results differ from the published values. These differences are documented rather than treated as exact reproductions.

Possible factors include implementation details, model checkpoints, configurations, and evaluation differences.

## Project Structure

```text
multilingual-text-detoxification/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── README.md
│   └── sample/
│
├── src/
│   ├── evaluation/
│   ├── baselines/
│   ├── models/
│   └── data/
│
├── predictions/
│
├── results/
│   ├── evaluation_results.csv
│   ├── results_analysis.md
│   ├── tables/
│   ├── per_example_metrics/
│   └── figures/
│
├── scripts/
│
└── docs/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-text-detoxification.git
cd multilingual-text-detoxification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Some model-based evaluation components may download pretrained checkpoints from Hugging Face during their first execution.

## Running the Evaluation

The evaluation pipeline accepts a test dataset and a prediction CSV.

Example:

```bash
python src/evaluation/evaluate.py \
    --test data/test.csv \
    --predictions predictions/backtranslation.csv \
    --model-name backtranslation \
    --batch-size 4
```

On Windows PowerShell, the same command can be run on one line:

```powershell
python src/evaluation/evaluate.py --test data/test.csv --predictions predictions/backtranslation.csv --model-name backtranslation --batch-size 4
```

Check the evaluation script and configuration file for the expected input columns and model requirements.

## Limitations

- The experiments do not constitute a complete reproduction of every experiment in the original COLING 2025 paper.
- GPT-4 was not evaluated in this project.
- Automatic evaluation metrics may not fully capture fluency, contextual appropriateness, or human-perceived toxicity.
- Some comparisons with published results are reference comparisons rather than exact checkpoint-to-checkpoint reproductions.
- The experiments focus primarily on the English evaluation setting documented here.
- Human evaluation would be valuable for validating the automatic metrics.

## Future Work

- Complete evaluation of preference-tuned detoxification models.
- Extend experiments to Hindi and Hinglish.
- Conduct human evaluation of generated sentences.
- Perform qualitative error analysis.
- Investigate additional transformer-based approaches.
- Study the effect of training-data quality and scale.
- Improve reproducibility through detailed experiment configurations.

## Research Reference

Dementieva et al. (2025).

**Multilingual and Explainable Text Detoxification with Parallel Corpora.**

Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025).

This project is inspired by the above research and distinguishes reproduction-oriented experiments from additional experiments and observations.
