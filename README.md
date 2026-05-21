# XGBoost Classifier — Diabetes Prediction

> Gradient boosting applied to the Pima Indians Diabetes dataset using the same battle-tested EDA pipeline as the Decision Tree and Random Forest projects — an `XGBClassifier(n_estimators=200, learning_rate=0.001)` built on a slow-learner strategy that combines many shallow corrections into a strong predictive ensemble.

---

## Problem

Predict whether a patient has diabetes based on diagnostic measurements. The same clinical context as the Decision Tree and Random Forest projects — this is the third model in a progression from a single interpretable tree, to a bagged forest, to a boosted ensemble. The central question: does gradient boosting's sequential error-correction add measurable value over bagging?

## Dataset

- **Source:** Pima Indians Diabetes Dataset (768 rows × 9 features)
- **Target:** `Outcome` — 1 = diabetes, 0 = no diabetes (65% / 35% class distribution)
- **Features used after cleaning:** Glucose, BMI, Age, Pregnancies (top 4 by SelectKBest f_classif)

The full EDA and preprocessing pipeline is identical to the Decision Tree and Random Forest projects:

| Step | Action |
|---|---|
| Drop impossible columns | Insulin (48.7% zeros) and SkinThickness (29.6% zeros) removed |
| Zero imputation | Group-stratified median imputation on Glucose, BloodPressure, BMI |
| Outlier capping | IQR method on Pregnancies, DiabetesPedigreeFunction, Age |
| Scaling | StandardScaler |
| Feature selection | SelectKBest (f_classif, k=4) → Glucose, BMI, Age, Pregnancies |
| Split | 80/20 stratified (614 train / 154 test) |

## Model

**XGBClassifier(n_estimators=200, learning_rate=0.001, random_state=42)**

XGBoost builds trees sequentially — each tree fits the residual errors of all the trees before it. This is fundamentally different from a Random Forest, where trees are independent and vote in parallel. With boosting, later trees specifically target the cases the model currently gets wrong.

The hyperparameter choices reflect a deliberate **slow learner strategy:**
- `n_estimators=200` — 200 sequential trees, enough depth to compound many small corrections
- `learning_rate=0.001` — each tree contributes only a tiny step toward the correct prediction, preventing any single tree from overfitting the residuals

This combination — many trees, each contributing very little — is a known regularisation approach in gradient boosting. The tradeoff: longer training time in exchange for smoother generalisation.

## The Boosting vs. Bagging Distinction

| Approach | Trees | How they combine | Target |
|---|---|---|---|
| Random Forest (bagging) | Parallel, independent | Majority vote | Reduce variance |
| XGBoost (boosting) | Sequential, dependent | Additive residual fitting | Reduce bias |

A Random Forest builds 60 diverse trees simultaneously and averages them to cancel noise. XGBoost builds 200 trees one at a time, each correcting what the previous model got wrong — progressively reducing the training error.

## Key Takeaways

- **Boosting corrects bias; bagging reduces variance:** A Random Forest averages independent trees to reduce prediction variance. XGBoost instead targets the samples the model consistently mispredicts, iteratively reducing bias. These are complementary strategies for different failure modes.
- **Learning rate and n_estimators are coupled hyperparameters:** A very low `learning_rate` (0.001) is only useful if `n_estimators` is high enough to compensate — each tree makes such a tiny contribution that hundreds are needed to accumulate a useful prediction. Tuning one without the other produces a suboptimal model.
- **XGBoost handles the same preprocessed features as simpler models:** Unlike neural networks, gradient boosting on tabular data typically works well with the same feature engineering pipeline used for Decision Trees and Logistic Regression — no architectural changes required.

## Tech Stack

`Python` · `xgboost` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_BoostingAlgorithms_MTK.git
cd ML_BoostingAlgorithms_MTK
pip install -r requirements.txt
jupyter notebook src/BoostRevised.ipynb
```

The trained model is saved to `models/` via `pickle`.

## What I'd Do Next

- Run `GridSearchCV` over `n_estimators`, `learning_rate`, `max_depth`, and `subsample` to find the optimal combination — the current slow-learner configuration is principled but not tuned
- Compare directly against the Random Forest (n_estimators=60, no tuning) and the optimised Decision Tree (max_depth=5) on the same test split to measure the actual accuracy gain from boosting
- Add **SHAP values** to explain individual predictions — XGBoost integrates natively with the SHAP library, enabling per-patient feature attribution without sacrificing model power

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
