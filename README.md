# Predicting Valorant Esports Map Outcomes Using Machine Learning

DS 4400, Machine Learning and Data Mining I, Spring 2026
Rohil Singh and Sean Tian

This repository contains the code, data, and final report for our DS 4400 project. We build two supervised learning setups for predicting the winner of a Valorant map. Task A is a same map baseline. Task B is the main project, which predicts the winner using only information available before the map starts.

## Repository layout

```
.
├── README.md
├── preliminary-work/            (work done to find leaky features and guide final)
├── notebooks/
│   ├── 01_build_datasets.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_ffnn_model.ipynb
│   └── 04_analysis_and_figures.ipynb
├── outputs/                     (all generated CSVs and PNG figures)
├── report/
│   └── DS4400_Valorant_Final_Report.docx
└── data/
    └── NewVLRDataRaw.csv        (raw VLR.gg match data, placed here for notebook 1)
```

## Run order

All four notebooks were developed on Google Colab with file upload prompts. Run in this order.

1. `01_build_datasets.ipynb`
   Upload the raw VLR CSV when prompted. Produces `taskA_dataset.csv`, `taskB_dataset.csv`, `dataset_summary.json`, `figure1_class_balance_and_years.png`, and `figure2_taskB_correlation_heatmap.png`.
2. `02_classical_models.ipynb`
   Upload `taskA_dataset.csv` and `taskB_dataset.csv`. Produces the classical model results and prediction CSVs for Logistic Regression, KNN, SVM, and Random Forest.
3. `03_ffnn_model.ipynb`
   Upload the two dataset CSVs. Produces the feedforward neural network results, loss plots, and predictions for both tasks.
4. `04_analysis_and_figures.ipynb`
   Upload the two dataset CSVs, all results CSVs, and all prediction CSVs. Produces the combined results table, corrected betting odds baseline, model comparison plots, ROC curves, permutation importance, minimal subset results, and the Task B FFNN confusion matrix and classification report.

## Data and methods summary

The VLR dataset used in this project is a publicly available collection of professional and semi professional Valorant match records. After cleaning, the working corpus contains 127,804 map level rows. We use a temporal train test split where matches before 2024 form the training set (108,358 rows) and matches from 2024 form the test set (19,446 rows). The binary target is `Team1 Win`.

For Task A we retained same map features after removing the most direct leakage columns. For Task B we constructed an entirely past only feature set that includes a simple Elo strength rating, rolling last three match averages for Rating, ACS, KAST, ADR, Kills, and Deaths, overall prior win rate, map specific prior win rate, head to head prior win rate, and a set of difference features such as Elo Diff and Recent Rating 3 Diff. All of these features were computed chronologically using the pandas shift and cumulative operations so that no value used as input for a given map could depend on the outcome of that same map.

We trained five models. Logistic Regression, K Nearest Neighbors, SVM, and Random Forest were implemented with scikit learn pipelines that use median imputation for numeric features, most frequent imputation for the categorical Map feature, one hot encoding for Map, and standard scaling for the numeric columns. The feedforward neural network was implemented in TensorFlow and Keras with a hidden 128 unit layer, dropout, a hidden 64 unit layer, dropout, and a single unit sigmoid output. For slower classical models on the full data we used a random training subsample of 30,000 to 40,000 rows, with evaluation always performed on the complete 19,446 row test set.

Evaluation uses six metrics, Accuracy, Precision, Recall, F1, ROC AUC, and Brier score. We also compute a corrected betting odds baseline using implied probability 1 divided by the decimal odds, on the 5,341 test maps where market odds were available.

## Note on Python style and course alignment

The notebooks follow the style of the course class notebooks.
- The feedforward neural network uses the Sequential plus list of layers pattern from the class NN notebook, prints `model.summary()` after construction, and uses a Keras training history plot for loss.
- The Random Forest uses the `max_depth`, `min_samples_leaf`, and `n_estimators` hyperparameters covered in the Decision Trees lecture.
- The Task B feature importance analysis uses Random Forest permutation importance, connecting to the variable importance framing in the bagging and Random Forests lecture.
- The best Task B model includes a confusion matrix and a per class classification report that mirror the diagnostic outputs used in the class Logistic Regression notebook.
- We use `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer` to chain imputation, scaling, and one hot encoding. This is the pattern recommended in the scikit learn documentation for mixed numeric and categorical data and is used here because our dataset has missing values and a categorical Map column, unlike the class demo datasets.

## Final results summary

Task A, best model (Logistic Regression), Accuracy 0.9963, F1 0.9968, ROC AUC 0.9997. All five models exceed 0.95 on Task A because the same map features effectively reconstruct the outcome.

Task B, best model (Feedforward Neural Network), Accuracy 0.6863, F1 0.7462, ROC AUC 0.7471. All five learned models exceed the corrected betting odds baseline (Accuracy 0.6141, F1 0.6681, ROC AUC 0.6592). The top three features by permutation importance are Elo Diff, Series Odds, and Team1 Map Odds. The minimal subset analysis shows that using only the top three features preserves most of the predictive performance of the full model.
