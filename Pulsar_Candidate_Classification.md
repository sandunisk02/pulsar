# Pulsar Candidate Classification Using Machine Learning Under Missing Data: A Comparative Analysis of Imputation Methods and Feature Importance

**Authors:** Shehan Nonis • Sanduni Kaveesha • Pumudu Ramuditha

---

## 1. Introduction: From Candidates to Pulsars

Radio surveys detect many candidate signals — but only a small fraction are genuine pulsars.

| Dataset summary | Value | Description |
|---|---:|---|
| Candidates | 17,898 | HTRU2 dataset observations |
| Pulsars | 1,639 | Positive / minority class |
| Positive rate | 9.16% | 1,639 of 17,898 observations |

### Why automation?

- Manual candidate inspection is costly at survey scale.
- Classification is a binary task: **pulsar (class 1)** versus **non-pulsar (class 0)**.
- A model must detect rare pulsars without generating excessive false alarms.

### Research focus

How does missing-data treatment affect Random Forest classification, and which features matter most?

---

## 2. Literature Survey

| Paper | What it addresses | Methods | Results |
|---|---|---|---|
| Lyon et al. (2016) | Pulsar candidate selection at scale | Historical review plus 8 new features and GH-VFDT classifier (Lyon et al., 2016) | Recall reached ~98% on test data; >90% in streams; false positives <0.5% (Lyon et al., 2016). |
| Breiman (2001) | General Random Forest classification | Ensemble of randomized decision trees with feature subsampling (Breiman, 1999) | Error depends on tree strength and correlation; robust to noise; variable importance available (Breiman, 1999). |
| Stekhoven & Bühlmann (2011) | Missing-value imputation for mixed-type data | Iterative RF imputation with OOB error estimation (Stekhoven & Bühlmann, 2011) | Handles mixed variables and tends to outperform alternatives when interactions are nonlinear (Stekhoven & Bühlmann, 2011). |
| Troyanskaya et al. (2001) | Missing microarray values | Compared KNNimpute, SVDimpute, and row average (Troyanskaya et al., 2001) | KNNimpute appeared more robust than SVDimpute; both beat row average and zeros (Troyanskaya et al., 2001). |
| Lyon (2017) | What comes next in candidate pulsar selection | Perspective on scaling, storage, and real-time ML selection (Lyon, 2017) | ML is described as the state of the art, but rising volumes still require evolution (Lyon, 2017). |

---

## 3. Problem Statement & Key Questions

**Problem:** Reliable pulsar-candidate classification when the working dataset contains missing values and severe class imbalance.

### Key questions

1. How does the imputation method affect classification performance?
2. Which features — or feature groups — provide the greatest predictive power?
3. Does class-weighted Random Forest improve minority-class recall and F-scores?
4. How consistently do the findings hold across stratified cross-validation folds?

---

## 4. Objectives & Hypotheses

### Objectives

1. Characterise missingness, feature distributions, and class imbalance.
2. Compare mean, median, KNN, and iterative Random-Forest imputation.
3. Identify informative features using feature-group and leave-one-feature-out tests.
4. Train and validate Random Forest (baseline and class-weighted), then apply the selected pipeline to the unlabeled test set.

### Hypotheses

**H1 — Imputation:** Performance will differ across imputation methods; if missingness is random, differences are expected to be small.

**H2 — Feature value:** Information is not uniform across the eight features; some will contribute more, and correlated features may share importance.

> These are hypotheses to test — not results already established.

---

## 5. Dataset: HTRU2

| Characteristic | Value |
|---|---:|
| Observations | 17,898 |
| Non-pulsars (class 0) | 16,259 |
| Pulsars (class 1) | 1,639 |
| Continuous features | 8 |

### Feature families

| Family | Number of features | Description |
|---|---:|---|
| Integrated pulse profile | 4 | Statistical summaries of the integrated pulse profile. |
| DM–SNR curve | 4 | Statistical summaries of the dispersion-measure / signal-to-noise curve. |

### Missing-value audit — training file (n = 12,528)

| Feature | Missing count | Missing (%) |
|---|---:|---:|
| Profile excess kurtosis | 1,735 | 13.85% |
| DM–SNR standard deviation | 1,178 | 9.40% |
| DM–SNR skewness | 625 | 4.99% |
| Other 5 features | 0 | 0.00% |

**Data integrity check:** Original HTRU2 has no missing values, so gaps were introduced in this project copy: 26% of training rows have at least one missing value, at similar rates in both classes.

---

## 6. Methodology: Imputation & Class Imbalance

### Compare imputation pipelines

| Method | Description |
|---|---|
| Mean | Simple baseline; sensitive to outliers. |
| Median | Robust central-value baseline. |
| KNN | Uses similar rows; scale features inside the pipeline. |
| Iterative RF | Predicts each gap from the other features, iteratively. |

### Compare imbalance strategies

| Strategy | Description |
|---|---|
| Baseline RF | No explicit class weighting. |
| Class-weighted RF | Penalise minority-class errors more strongly. |

**Leakage-safe rule:** Fit imputer and model on each training fold only. Apply the fitted imputer to that fold’s validation data.

### Controls to keep comparisons fair

- Use the same stratified folds for every configuration.
- Keep preprocessing inside a pipeline; never impute before splitting.
- Use a fixed random seed and record model hyperparameters.
- Methodology II uses one fixed imputer, chosen from the imputation results.

---

## 7. Methodology: Feature Analysis

### Feature-set comparisons

| Feature set | Description |
|---|---|
| All 8 features | Combined pulse-profile + DM–SNR features. |
| Profile only | Four integrated-profile features. |
| DM–SNR only | Four DM–SNR curve features. |

### Feature importance: Leave-One-Feature-Out (LOFO)

1. Fit the full model.
2. Remove one feature.
3. Refit and evaluate.
4. Compare the score drop.

**Interpretation:** A larger performance decrease after removal suggests greater predictive contribution; correlated features can share or mask importance.

---

## 8. Methodology: Validation & Metrics

### Stratified 5-fold cross-validation

The training portion is divided into five folds, each accounting for **20%**.

- Each run trains on **4 folds (80%)** and validates on the held-out fold **(20%)**.
- Rotate the validation fold; report **mean ± standard deviation** across the five runs.
- **Positive class = pulsar.** Choose the final configuration from cross-validation only.
- The supplied test file is **unlabelled**, so a **stratified 20% hold-out** from the labelled data stays untouched for one final evaluation.

### Metrics to report

| Metric | Why it matters |
|---|---|
| Precision | Of predicted pulsars, how many are pulsars? |
| Recall | Of real pulsars, how many are detected? |
| F1-score | Harmonic mean of precision and recall. |
| F2-score | Weights recall more heavily than precision. |
| Confusion matrix | Shows TP, FP, TN, and FN. |
| Accuracy | Overall correctness; interpret cautiously. |
| Average precision | Area under the precision–recall curve (PR-AUC). |

---

## 9. End-to-End Experimental Workflow

1. **Audit:** Missingness, duplicates, labels, and class balance.
2. **Split:** Hold out 20% (stratified); make CV folds from the rest.
3. **Preprocess:** Fit each imputer on the training fold only.
4. **Train:** Baseline RF and class-weighted RF.
5. **Compare:** Metrics and feature-group / LOFO analysis.
6. **Select:** Choose the pipeline from cross-validation results.
7. **Predict:** Refit on labelled data; predict the unlabeled test set.

**Deliverables:** Reproducible comparison table • feature-contribution summary • final unlabeled-set predictions.

---

## 10. References

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.
- Josse, J., Prost, N., Scornet, E., & Varoquaux, G. (2019). On the consistency of supervised learning with missing values. arXiv:1902.06931.
- Lee, E., Rustam, F., Aljedaani, W., Ishaq, A., Rupapara, V., & Ashraf, I. (2021). Predicting pulsars from imbalanced dataset with hybrid resampling approach. *Advances in Astronomy, 2021*, 4916494.
- Lin, H., & Li, X. (2023). Dealing with the data imbalance problem on pulsar candidates sifting based on feature selection. arXiv:2312.16366.
- Lyon, R. J. (2015). HTRU2 [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DK6R
- Lyon, R. J., Stappers, B. W., Cooper, S., Brooke, J. M., & Knowles, J. D. (2016). Fifty years of pulsar candidate selection: From simple filters to a new principled real-time classification approach. *Monthly Notices of the Royal Astronomical Society, 459*(1), 1104–1123.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.
- Stekhoven, D. J., & Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. *Bioinformatics, 28*(1), 112–118.
- Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., & Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics, 9*, 307.
- Troyanskaya, O., et al. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics, 17*(6), 520–525.
- Wang, Y., Pan, Z., Zheng, J., Qian, L., & Li, M. (2019). A hybrid ensemble method for pulsar candidate classification. arXiv:1909.05303.

---

## Appendix A: Evaluation Metrics

### Confusion matrix (pulsar = positive class)

| Actual class | Predicted pulsar | Predicted non-pulsar |
|---|---|---|
| Actual pulsar | True positive (TP) | False negative (FN) |
| Actual non-pulsar | False positive (FP) | True negative (TN) |

### Core definitions

- **Precision:** $TP / (TP + FP)$
- **Recall:** $TP / (TP + FN)$
- **F1:** $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$
- **F2:** $5 \times (\text{Precision} \times \text{Recall}) / (4 \times \text{Precision} + \text{Recall})$
- **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$

### Reporting recommendation

Report **mean ± standard deviation** across folds. Include the confusion matrix and, where possible, a precision–recall curve / average precision. State the decision threshold and positive class explicitly.

> **Caution:** Avoid claiming a best method before running the experiment; the presentation describes a **study plan**, not experimental findings.

---

## Appendix B: 5-Fold Cross-Validation Schematic

Each fold serves as validation once; the remaining four folds train the model.

| Run | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| 1 | **VALIDATE** | Train | Train | Train | Train |
| 2 | Train | **VALIDATE** | Train | Train | Train |
| 3 | Train | Train | **VALIDATE** | Train | Train |
| 4 | Train | Train | Train | **VALIDATE** | Train |
| 5 | Train | Train | Train | Train | **VALIDATE** |

Use **stratification** so each fold approximately preserves the pulsar / non-pulsar class proportions.

All preprocessing (including imputation) is fitted independently within each training fold.
