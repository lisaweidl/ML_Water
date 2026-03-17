# ML_Water

Machine-learning workflow developed as part of a master’s thesis on groundwater monitoring effectiveness in the Austrian Bohemian Massif.

## Project Background

This repository contains the data-processing and machine-learning workflow used in the thesis:

**“Evaluating Groundwater Monitoring Effectiveness and Adaptation Needs in the Austrian Bohemian Massif: A Focus on Spatial Arrangement and Climate Preparedness.”**

The empirical focus is the groundwater body group **GK100094** in the **Waldviertel region of Lower Austria**. The broader thesis examines how far existing monitoring data support the interpretation of groundwater quality and quantity, predictive analysis, and governance-related reporting. In this repository, the emphasis is on the **diagnostic machine-learning component** of that work. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

## Project Objective

The goal of this repository is to test whether the available groundwater monitoring data contain a stable predictive structure that can be used to estimate **orthophosphate** concentrations.

The machine-learning analysis is designed as a **diagnostic tool**, not as a full mechanistic groundwater model. In other words, model performance is used to assess whether the monitoring record supports meaningful prediction under real-world constraints such as sparse measurements, uneven temporal coverage, and limited overlap between groundwater and weather records. :contentReference[oaicite:5]{index=5}

## Research Focus

The modeling work addresses three main questions:

- Can orthophosphate be predicted from the available monitoring data?
- Do machine-learning models outperform simple statistical baselines?
- Do meteorological predictors improve predictive performance?

According to the thesis, the models show that the dataset contains **learnable structure** and can outperform simple baselines, but **meteorological predictors add little value** because the overlap between groundwater and weather records is limited in space and time. :contentReference[oaicite:6]{index=6}

## Target Variable

The primary prediction target is:

- **Orthophosphate (PO₄³⁻)**

Orthophosphate was chosen because it is one of the parameters contributing to the poor chemical status of GK100094 and is therefore highly relevant for groundwater-quality interpretation in the study area. Another parameter, a dimethachlor metabolite, was considered initially but excluded from supervised learning because the available observations were too sparse for reliable modeling. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

## Methods Overview

The workflow implemented in this repository follows the thesis methodology:

1. **Data cleaning**
   - preparation of groundwater and weather datasets
   - handling of missing values
   - formatting and harmonization of variables

2. **Feature engineering**
   - creation of lag features from past groundwater observations
   - creation of rolling-window summaries for selected meteorological variables
   - encoding of site identifiers for pooled modeling across monitoring sites

3. **Model training**
   - **Random Forest regression**
   - **k-nearest neighbours (k-NN) regression**

4. **Evaluation**
   - train/test split
   - cross-validation within the training set
   - hyperparameter tuning
   - comparison against simple baseline predictors
   - evaluation using **R², MAE, and RMSE**

The thesis also emphasizes that interpretation of feature importance is **diagnostic rather than causal**. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

## Models Used

### Random Forest
Random Forest is used because it can handle nonlinear relationships, correlated predictors, noisy environmental data, and heterogeneous behavior across monitoring locations. It also allows diagnostic interpretation through variable-importance analysis. :contentReference[oaicite:11]{index=11}

### k-Nearest Neighbours
k-NN is used as a complementary non-parametric model to test whether orthophosphate can be approximated through local similarity in predictor space. Because it is distance-based, feature scaling is important. :contentReference[oaicite:12]{index=12}

## Feature Engineering

The thesis describes two main feature types:

- **Lag features (1–3 previous observations)** to incorporate recent groundwater history
- **Rolling-window features (7, 30, and 90 days)** for selected meteorological variables

This allows the models to use temporal structure while avoiding information leakage by restricting feature construction to current and past information only. :contentReference[oaicite:13]{index=13}

## Evaluation Strategy

To test generalization, the data are split into training and test sets, with model tuning and cross-validation performed only on the training subset. Final performance is reported on the held-out test data using:

- **R²**
- **MAE**
- **RMSE**

Model performance is also compared with simple baseline predictors to assess whether machine learning adds value beyond central tendency or persistence. :contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15}

## Repository Purpose

This repository is intended to support:

- transparent documentation of the thesis modeling workflow
- reproducibility of the machine-learning analysis
- inspection of data-cleaning, preprocessing, and feature-engineering choices
- comparison of ML performance under real monitoring-data limitations

## Main Takeaway

The machine-learning results suggest that the available groundwater monitoring data in GK100094 contain a meaningful predictive signal for orthophosphate. However, the thesis concludes that the added value of meteorological predictors is limited because groundwater, weather, precipitation, and related records are not consistently aligned across the monitoring network. :contentReference[oaicite:16]{index=16}

## Thesis Reference

This repository accompanies the master’s thesis by **Lisa-Marie Weidl**:

*Evaluating Groundwater Monitoring Effectiveness and Adaptation Needs in the Austrian Bohemian Massif: A Focus on Spatial Arrangement and Climate Preparedness.* :contentReference[oaicite:17]{index=17}
