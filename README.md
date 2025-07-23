# Custom Sentiment Analysis: Electronics Product Reviews

This project is a sentiment analysis pipeline built using transformer models (BERT, RoBERTa, DistilBERT). The goal is to classify customer reviews of electronics products into three categories: **negative**, **neutral**, and **positive**.

---

## Project Overview

The task was part of a Machine Learning interview challenge to:
- Perform **zero-shot** classification using `facebook/bart-large-mnli`
- **Fine-tune** a transformer model on labeled sentiment data
- Evaluate performance on validation and test sets
- Conduct **error analysis**
- Present results with classification reports, charts, and tables

---

## File Structure

| File | Description |
|------|-------------|
| `Welcome_To_Colab.ipynb` | Main notebook: preprocessing, model training, evaluation, and visualization |
| `train.csv` / `test.csv` / `valid.csv` | Labeled datasets used for training and evaluation |
| `README.md` | This project summary file |

---

## Key Features

- **Preprocessing**: Cleaned and tokenized review data
- **Zero-Shot Evaluation**: Performed using `facebook/bart-large-mnli`
- **Fine-Tuning**: Performed using a HuggingFace transformer model
- **Visualization**: F1-score tables and accuracy bar charts
- **Error Analysis**: Identified and analyzed incorrect predictions

---

## Results

- **Validation Accuracy**: 100%
- **Test Accuracy**: 100%
- All classes achieved **perfect precision, recall, and F1-scores**

---

## Error Analysis

Even though the model achieved 100% accuracy, error analysis was still conducted to inspect predicted vs actual mismatches, if any.

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Google Colab
- Matplotlib / Seaborn
- Pandas / NumPy

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the dataset CSV files
3. Run cells sequentially to:
   - Load data
   - Train/fine-tune model
   - Generate metrics
   - Visualize results
   - Perform error analysis

---

## Author

- **Name:** Perry Mason
- **GitHub:** masonperry248
- **Email:** masonperry248@gmail.com

---
