# Pulsar Candidate Classification Using Machine Learning

A machine-learning study of pulsar-candidate classification using the
HTRU2 dataset. The project investigates how missing-value imputation and
class-imbalance handling affect the detection of genuine pulsars, and
examines the predictive contribution of different signal features.

## Project overview

Radio-pulsar surveys produce large numbers of candidate signals, while
only a small fraction are genuine pulsars. This project treats candidate
selection as a binary classification problem:

-   **Class 1:** Pulsar
-   **Class 0:** Non-pulsar

The main focus is a comparison of four imputation approaches, two Random
Forest class-weighting strategies, feature-family comparisons, and
leave-one-feature-out (LOFO) analysis.

## Dataset

The project uses the **HTRU2** pulsar-candidate dataset.

  Dataset characteristic                                        Value
  ------------------------------------------- -----------------------
  Total observations                                           17,898
  Pulsars (class 1)                                             1,639
  Non-pulsars (class 0)                                        16,259
  Features                                      8 continuous features
  Training rows used in the notebook                           12,528
  Unlabelled test rows used in the notebook                     5,370

The eight features are grouped into two families:

1.  **Integrated pulse profile (4 features):** mean, standard deviation,
    excess kurtosis, and skewness.
2.  **DM--SNR curve (4 features):** mean, standard deviation, excess
    kurtosis, and skewness.

The project copy contains missing values in three features. In the
training file, the missing-value counts are:

  Feature                                       Missing values   Missing (%)
  ------------------------------------------- ---------------- -------------
  Excess kurtosis of the integrated profile              1,735        13.85%
  Standard deviation of the DM-SNR curve                 1,178         9.40%
  Skewness of the DM-SNR curve                             625         4.99%

The notebook checks the input data, missing values, duplicate rows, and
class distribution before modelling.

> **Data note:** The notebook expects `pulsar_data_train.csv` and
> `pulsar_data_test.csv`. These data files are not included in the
> repository structure shown here. Obtain the dataset files separately
> and place them in the expected location before running the notebook.

## Repository structure

``` text
sandunisk02-pulsar/
├── Pulsar_Candidate_Classification.md
├── Pulsar_Classification_Complete_Steps.ipynb
├── pyproject.toml
└── README.md
```

-   `Pulsar_Candidate_Classification.md` --- project background,
    research questions, hypotheses, methodology, evaluation plan, and
    references.
-   `Pulsar_Classification_Complete_Steps.ipynb` --- end-to-end
    practical notebook, including data inspection, exploratory analysis,
    model comparison, feature analysis, holdout evaluation, and test-set
    prediction.
-   `pyproject.toml` --- Python version requirement and project
    dependencies.
-   `README.md` --- project overview and instructions.

## Research questions

1.  How does the imputation method affect classification performance?
2.  Which individual features or feature groups provide the greatest
    predictive information?
3.  Does class-weighted Random Forest improve recall and F-scores for
    the minority pulsar class?
4.  How consistent are the results across stratified cross-validation
    folds?

The project tests two hypotheses:

-   **H1 --- Imputation:** Classification performance may differ between
    imputation methods. If missingness is random, the differences may be
    small.
-   **H2 --- Feature value:** Predictive information may not be
    distributed equally across the eight features; correlated features
    may share or mask importance.

These are hypotheses investigated by the experiment, not assumptions
that every result must support.

## Methodology

### 1. Missing-value imputation

Four methods are compared:

  -----------------------------------------------------------------------
  Method                              Description
  ----------------------------------- -----------------------------------
  Mean                                Replaces missing values with the
                                      feature mean.

  Median                              Replaces missing values with the
                                      feature median.

  KNN                                 Estimates missing values from
                                      similar observations; features are
                                      standardised for distance
                                      calculations.

  Iterative Random Forest             Iteratively predicts missing values
                                      using the other features and a
                                      Random Forest regressor.
  -----------------------------------------------------------------------

### 2. Class-imbalance strategies

Each imputation method is evaluated with:

-   **Baseline Random Forest:** no explicit class weighting.
-   **Class-weighted Random Forest:** uses `class_weight="balanced"` to
    give greater weight to the minority class.

### 3. Feature analysis

The notebook compares:

-   All eight features.
-   The four integrated pulse-profile features.
-   The four DM--SNR curve features.

It also performs **leave-one-feature-out (LOFO)** analysis. One feature
is removed at a time, the model is retrained, and the change in
cross-validation F2-score is measured. A larger positive score drop
indicates a stronger predictive contribution in this experiment.
Correlations between features can affect this interpretation.

### 4. Validation and leakage prevention

-   A stratified 20% holdout is separated from the labelled training
    data and kept untouched during model and imputer selection.
-   The remaining 80% is used for stratified 5-fold cross-validation.
-   Imputation and modelling are performed inside a scikit-learn
    pipeline, so preprocessing is fitted only on each training fold.
-   The same folds and random seed (`42`) are used for fair comparisons.
-   The configuration is selected using mean cross-validation F2-score,
    prioritising detection of the minority pulsar class.
-   The selected pipeline is evaluated once on the untouched holdout.
-   Finally, the selected pipeline is refitted on all labelled training
    rows and applied to the unlabelled test data.

### Evaluation metrics

The notebook reports:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   F2-score
-   Average precision (area under the precision--recall curve)
-   Confusion matrix

Because pulsars are the minority class, accuracy should not be
interpreted on its own. The positive class is **pulsar (class 1)**.

## Results included in the notebook

The following values are the outputs recorded in the supplied notebook.
They describe this run and should not be treated as universal
performance guarantees.

### Imputation and class-weighting comparison

The best configuration by the notebook's stated selection criterion
(mean cross-validation F2-score) was **KNN imputation with
class-weighted Random Forest**.

  Imputer          Class weighted?   Mean CV F2
  -------------- ----------------- ------------
  KNN                          Yes       0.8676
  Iterative RF                 Yes       0.8618
  Median                       Yes       0.8589
  Mean                         Yes       0.8582
  KNN                           No       0.8434
  Iterative RF                  No       0.8419
  Median                        No       0.8402
  Mean                          No       0.8402

For the selected configuration:

-   Mean CV F2-score: **0.8676 ± 0.0184**
-   Mean CV recall: **0.8655**
-   Mean CV precision: **0.8774**
-   Mean CV average precision: **0.9159**

### Feature-family comparison

  Feature set                Mean CV F2
  ------------------------ ------------
  All 8 features                 0.8676
  Integrated profile (4)         0.8581
  DM--SNR curve (4)              0.6892

In this run, the full eight-feature set achieved the highest mean
cross-validation F2-score.

### Untouched holdout evaluation

The selected pipeline was evaluated on the reserved 20% holdout:

  Metric                Holdout result
  ------------------- ----------------
  Accuracy                      0.9800
  Precision                     0.8918
  Recall                        0.8918
  F1-score                      0.8918
  F2-score                      0.8918
  Average precision             0.9326

Confusion matrix (rows: actual class; columns: predicted class):

``` text
                 Predicted
                 Non-pulsar  Pulsar
Actual non-pulsar    2250       25
Actual pulsar          25      206
```

### Unlabelled test-set predictions

The notebook generated predictions for **5,370** unlabelled test
observations:

-   Predicted non-pulsars: **4,904**
-   Predicted pulsars: **466**

The supplied test file has no ground-truth labels, so accuracy or other
supervised test metrics cannot be calculated for it.

## Setup

The project metadata in `pyproject.toml` specifies **Python 3.13 or
later** and the following dependencies:

-   `ipykernel`
-   `joblib`
-   `matplotlib`
-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `seaborn`

Create and activate a virtual environment, then install the project:

``` bash
python -m venv .venv
```

On Windows:

``` powershell
.venv\Scripts\activate
```

On macOS or Linux:

``` bash
source .venv/bin/activate
```

Install the project and dependencies:

``` bash
pip install .
```

If you prefer to install the notebook dependencies directly:

``` bash
pip install ipykernel joblib matplotlib numpy pandas scikit-learn seaborn
```

## Input data and running the notebook

1.  Obtain the HTRU2 training and test CSV files used by this project.
2.  Name them:
    -   `pulsar_data_train.csv`
    -   `pulsar_data_test.csv`
3.  Make sure the CSV files are in the paths expected by the notebook.
4.  Open `Pulsar_Classification_Complete_Steps.ipynb` in Jupyter
    Notebook, JupyterLab, or Google Colab.
5.  Run the notebook cells from top to bottom.

**Check the data paths before running:** the notebook's introductory
text says to place the CSV files in the same folder as the notebook,
while the loading code currently reads from
`../Data/pulsar_data_train.csv` and `../Data/pulsar_data_test.csv`.
Either create the `Data` directory one level above the notebook with
those files, or edit the two `pd.read_csv(...)` paths to match where you
placed them.

The notebook uses `random_state=42` for reproducibility. The iterative
Random Forest imputation configuration may take longer to run than the
simpler imputers.

## Generated outputs

Running the notebook writes the following files to the notebook's
working directory:

  -------------------------------------------------------------------------
  Output file                           Description
  ------------------------------------- -----------------------------------
  `imputation_model_comparison.csv`     Cross-validation comparison of
                                        imputation and class-weighting
                                        configurations.

  `feature_group_comparison.csv`        Performance comparison of feature
                                        families.

  `leave_one_feature_out.csv`           LOFO feature-contribution results.

  `holdout_evaluation.csv`              Metrics from the untouched holdout
                                        evaluation.

  `pulsar_test_predictions.csv`         Test-set predictions and estimated
                                        pulsar probabilities.

  `pulsar_random_forest_model.joblib`   Saved fitted Random Forest
                                        pipeline.
  -------------------------------------------------------------------------

The notebook also displays exploratory plots, a holdout confusion
matrix, a precision--recall curve, and a LOFO feature-contribution plot.

## Important implementation note

The notebook implements iterative imputation using scikit-learn's
`IterativeImputer` with a `RandomForestRegressor`. This is an
approximation to Random-Forest-based iterative imputation; it does
**not** claim to reproduce the original `missForest` package exactly.

## References

The project background document contains the literature review and full
reference list. Key references include:

-   Breiman, L. (2001). Random forests. *Machine Learning, 45*(1),
    5--32.
-   Lyon, R. J., et al. (2016). Fifty years of pulsar candidate
    selection: From simple filters to a new principled real-time
    classification approach. *Monthly Notices of the Royal Astronomical
    Society, 459*(1), 1104--1123.
-   Stekhoven, D. J., & Bühlmann, P. (2012). MissForest---non-parametric
    missing value imputation for mixed-type data. *Bioinformatics,
    28*(1), 112--118.
-   Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is
    more informative than the ROC plot when evaluating binary
    classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.
-   Lyon, R. J. (2015). HTRU2 \[Data set\]. UCI Machine Learning
    Repository. https://doi.org/10.24432/C5DK6R

See
[`Pulsar_Candidate_Classification.md`](Pulsar_Candidate_Classification.md)
for the complete literature survey, research hypotheses, methodology
description, and references.

## Authors

-   Shehan Nonis
-   Sanduni Kaveesha
-   Pumudu Ramuditha
