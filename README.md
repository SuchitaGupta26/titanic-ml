
# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster using a Random Forest classifier.

Built as part of learning the end-to-end ML pipeline: data loading, exploratory data analysis, feature engineering, model training, and evaluation.

---

## Results

| Metric | Score |
|--------|-------|
| Model | Random Forest (100 trees) |
| Accuracy | 81% |
| Training samples | 712 |
| Test samples | 178 |

---

## Project structure

```
titanic-ml/
├── titanic_survival.ipynb   # Main notebook (run this)
└── README.md
```

---

## What the notebook covers

**Step 1 — Load data**
Dataset loaded directly from URL — no Kaggle account needed.

**Step 2 — Exploratory data analysis (EDA)**
- Survival rates by gender and passenger class
- Distribution of age and fare
- Missing value analysis

**Step 3 — Data cleaning & feature engineering**
- Filled missing `Age` values with median imputation
- Engineered two new features: `FamilySize` (SibSp + Parch + 1) and `IsAlone`
- Encoded categorical columns: `Sex` and `Embarked`

**Step 4 — Model training**
- 80/20 train-test split with `random_state=42` for reproducibility
- Trained `RandomForestClassifier` with 100 estimators

**Step 5 — Evaluation**
- Accuracy score, classification report (precision, recall, F1)
- Confusion matrix
- Feature importance plot

---

## Feature importance

| Feature | Importance |
|---------|-----------|
| Sex (gender) | 28% |
| Fare | 22% |
| Age | 18% |
| Pclass | 14% |
| FamilySize | 10% |
| Embarked | 5% |
| IsAlone | 3% |

Gender and ticket fare were the strongest predictors of survival — consistent with the historical record ("women and children first").

---

## How to run

**Option 1 — Google Colab (recommended, free)**

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `titanic_survival.ipynb`
3. Run all cells: Runtime → Run all

**Option 2 — Local**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook titanic_survival.ipynb
```

---

## Libraries used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualizations
- `scikit-learn` — model training and evaluation

---

## Key learnings

- How to handle missing data using median imputation
- How to encode categorical variables (label encoding)
- How to engineer new features from existing columns
- How to interpret a confusion matrix and classification report
- How Random Forest builds multiple decision trees to reduce overfitting

---

## Dataset

[Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — 891 passengers, 12 columns. Originally from Kaggle's Titanic competition.
