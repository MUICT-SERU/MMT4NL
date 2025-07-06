# MMT4NL
A repository for the paper "Test It Before You Trust It: Applying Software Testing for Trustworthy In-context Learning."
This repository contains scripts, datasets, prompts, and results for running and evaluating prompt-based experiments, including question answering (QnA) and sentiment analysis, with and without context.

## Folder Structure

```
Experiment/
│
├── readme.md
├── Datasets/
├── Prompts/
├── Results/
└── Scripts/
```

---

### 1. `Datasets/`

Contains raw and processed datasets used for experiments.

- `chat_dataset.csv`, `clean_qna_dataset.csv`: CSV files with QnA data.
- `strategyqa_train.json`: JSON dataset for QnA tasks.
- `Datasets.md`: Documentation about datasets.

---

### 2. `Prompts/`

Contains prompt templates for different tasks and settings.

- `qna_with_context/`: Prompts for QnA tasks with context (e.g., coreference, fairness, negation, robustness).
- `qna_without_context/`: Prompts for QnA tasks without context.
- `sentiment/`: Prompts for sentiment analysis.

---

### 3. `Results/`

Stores outputs and evaluation results.

- `qa_with_context/`: Results for QnA with context.
- `qa_without_context/`: Results for QnA without context.
- `sentiment_result/`: Results for sentiment analysis.

---

### 4. `Scripts/`

Contains all code and notebooks for running experiments.

- `01_sentiment_notebook.ipynb`: Sentiment analysis experiments.
- `02_qna_no_context_notebook.ipynb`: QnA without context experiments.
- `03_qna_with_context_notebook.ipynb`: QnA with context experiments.
- `PromptOps/`: Python package with utility modules (e.g., template formatters, perturbation, test suite).

---

## Getting Started

1. **Datasets:** Place or update datasets in the `Datasets/` folder.
2. **Prompts:** Edit or add prompt templates in the `Prompts/` subfolders.
3. **Scripts:** Run the notebooks in `Scripts/` to generate prompts, run models, and evaluate results.
4. **Results:** Find generated outputs and evaluation metrics in the `Results/` folders.

---

## Notes

- Update API keys and file paths in scripts as needed.
