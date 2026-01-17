# Emotion Detection AML

Multiclass emotion classification project comparing **DistilBERT** (transformer-based) and **LSTM** (recurrent neural network) approaches on an emotion-labeled text dataset.

Goal: Build reliable emotion detection models with strong class separability, good handling of ambiguous cases, and interpretable predictions (using LIME + word clouds).

## Project Overview

This repository contains notebooks for:
- Data preprocessing and exploration
- Training & evaluation of two main models:
  - DistilBERT (fine-tuned transformer)
  - LSTM (with word embeddings)
- Model interpretability analysis (LIME explanations + word clouds)
- Threshold tuning experiments for improved recall

## Notebooks

- `Emotions_Data_Preprocessing.ipynb`  
  Data cleaning, exploration and preprocessing

- `initial_DistilBERT_model.ipynb`  - https://colab.research.google.com/drive/1E1ghNoUmzLWHfEesiEULqPn9oP8i3AHs?usp=sharing
  First version of DistilBERT training and evaluation

- `retraining_DistilBERT.ipynb`  
  Updated DistilBERT training + LIME explanations + Word Cloud visualization

- `LSTM_Model.ipynb`  
  Original LSTM baseline model

- `retraining_LSTM.ipynb`  
  Retrained and improved LSTM version
