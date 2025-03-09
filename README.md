# AI-Generated Text Detector

A transformer-based classifier distinguishing between human-authored and AI-generated texts, fine-tuned on extensive academic and diverse-domain datasets. Built on DistilBERT, enhanced by linguistic features, perplexity measures, and sentiment analysis for robust interpretability.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Training](#model-training)
    - [Fine-Tuning Approach](#fine-tuning-approach)
- [Evaluation Metrics](#evaluation-metrics)
    - [Performance Results](#performance-results)
    - [Confusion Matrix](#confusion-matrix)
- [Usage](#usage)
- [Installation](#installation)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

This project leverages DistilBERT fine-tuned on ~23K labeled examples sourced from Kaggle's academic essays dataset, alongside the L2R dataset, which spans domains like academic research, business, and legal documents. The model utilizes sophisticated linguistic indicators and ensemble-based cross-validation for reliable identification of AI-generated content.

---

## Features

- **Transformer-based Detection**: DistilBERT fine-tuned specifically for AI-text discrimination.
- **Advanced Linguistic Indicators**:
    - **Readability**: Flesch-Kincaid, average sentence length.
    - **Structural Features**: Lexical diversity, burstiness (sentence length variance), function-word and stop-word ratios.
    - **Semantic Indicators**: TF-IDF cosine similarity, approximate edit distance.
    - **Perplexity Scoring**: GPT-2-based perplexity (higher indicates human-like complexity).
    - **Sentiment Analysis**: VADER sentiment (positive, neutral, negative thresholds).
- **Multi-format File Analysis**: Text, PDF, LaTeX, DOCX support.
- **CLI & PDF Reporting**: Command-line interface and comprehensive PDF reports detailing overall and sentence-level analyses.

---

## Dataset

- **Total Size**: 23,782 examples
- **Data Sources**:
    - **Kaggle Academic Essays Dataset**: Human-written academic texts and GPT-generated counterparts.
    - **Learning to Rewrite (L2R)**: Extensive dataset spanning domains such as Academic Research, Legal, Business, and Creative Writing.

- **Preprocessing**:
    - Text normalization, markup removal (LaTeX/Markdown), tokenization.
    - Computation of linguistic and statistical features for interpretability.

---

## Model Training

### Fine-Tuning Approach

- **Base Model**: Hugging Face `distilbert-base-uncased`
- **Training Configuration**:
    - **Epochs**: 3 per fold
    - **Batch Size**: 8 (optimized for Apple Silicon M2 GPU via Metal Performance Shaders)
    - **Learning Rate**: 4.67e-5 with linear decay scheduler
    - **Optimizer**: AdamW
    - **Hardware Acceleration**: Apple Metal Performance Shaders (MPS)

- **Cross-Validation Strategy**:  
  Implemented 5-fold cross-validation for robust evaluation, ensuring generalizability and providing a solid foundation for ensemble predictions.

---

## Evaluation Metrics

Model performance is rigorously assessed via:

- **Accuracy**: Overall classification correctness.
- **Precision**: Reliability of positive predictions (AI-generated text).
- **Recall**: Proportion of actual AI texts accurately classified.
- **F1-score**: Balance between precision and recall.
- **Confusion Matrix**: Visualization of True Positives, False Positives, True Negatives, and False Negatives.

### Performance Results

**Aggregated Cross-Validation Performance (5-folds):**

| Metric                 | Human-Written (%) | AI-Generated (%) |
|------------------------|-------------------|------------------|
| **Precision**          | 94.60             | 93.44            |
| **Recall**             | 83.91             | 97.97            |
| **F1-score**           | 88.91             | 95.65            |
| **Overall Accuracy**   | 93.82             |                  |

*(Metrics averaged over 5-fold cross-validation.)*

### Confusion Matrix

Aggregated confusion matrix across 5-fold cross-validation:

| Actual \ Predicted | Human-Written | AI-Generated |
|--------------------|---------------|--------------|
| **Human-Written**  | 5,816 (83.91%)| 1,098 (16.09%)|
| **AI-Generated**   | 330 (2.03%)   | 15,955 (97.97%)|

The confusion matrix indicates excellent recall in detecting AI-generated texts with ongoing efforts to minimize false positives (human texts identified as AI-generated).

---

## Usage

### Command-Line Interface

Rapid analysis of documents:

```bash
python ai_text_detector.py "/path/to/file.pdf"