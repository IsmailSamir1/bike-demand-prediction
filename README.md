# 🚲 Rush Hour: Bike-Sharing Demand Prediction

Predicting hourly bike-rental demand with **linear models built from scratch** and then improved step by step: gradient descent, polynomial features, bias–variance diagnosis, regularization (L1/L2/Elastic Net) and logistic regression. Each phase hands its output to the next one.

Machine Learning course project, German International University (GIU), Cairo. Team of 3.

## Highlights

- **Gradient descent from scratch in NumPy.** The stability limit came from the loss curvature: the largest eigenvalue of (2/n)XᵀX is 4.82, so α must stay below 2/4.82 ≈ 0.415. A learning-rate sweep confirmed it (α = 0.45 diverged and α = 0.4 zig-zagged). We used α = 0.3.
- **Feature engineering raised validation R² from 0.69 to 0.84.** One-hot hours plus *hour × working-day* interactions let the model learn separate commute and weekend curves.
- **Asymmetric loss (bonus).** Under-predicting costs 3× as much as over-predicting. The gradient was derived by hand and checked by finite differences (error 1.5e-8). This cut the operator's cost by **18%**.
- **Honest evaluation.** A chronological split shows the true error on unseen time periods, and bootstrap confidence intervals decide which improvements are real and which are noise.
- **A simple rule beat regularization.** Clipping predictions at 0 (rentals can't be negative) gained more accuracy than any regularization penalty.

## Results

| Phase | Model | Validation R² | Validation RMSE |
|---|---|---|---|
| Baseline | Predict the mean | – | ≈181 |
| 1 | Linear regression (gradient descent) | 0.694 | 100.9 |
| 2 | Polynomial degree 4 + hour × working-day | 0.837 | 73.6 |
| 4 | Lasso (λ = 0.316, 74 of 82 features) | 0.836 | 73.8 |
| Final | Lasso + clip at 0 | **0.855** | **69.5** |

**High-demand classifier (Phase 5).** A logistic regression with ROC-AUC **0.845** and F1 0.63 at a tuned cutoff of 0.30. It catches 78% of high-demand hours.

## Approach

| Phase | What we did | Key finding |
|---|---|---|
| 0. Data prep | Seeded 80/20 split. Repaired broken sensor readings (atemp, humidity). One-hot encoded categorical codes. Fit every transform on training rows only | Hour as one number gives R² 0.41; one-hot gives 0.69 |
| 1. Gradient descent | MSE, update w ← w − α·(2/n)Xᵀ(Xw − y), relative-improvement stopping rule | Converged to within 0.02% of the exact least-squares solution |
| 2. Polynomial regression | Powers of temp/humidity and hour × working-day products, warm-started from Phase 1 | Degree 4 chosen by bootstrap. Interactions matter far more than powers |
| 3. Bias–variance | Degrees 1–8, learning curves, chronological split | Bias-limited, not overfit. Unseen periods cost about 8 more bikes of error |
| 4. Regularization | Ridge, Lasso and Elastic Net over a shared λ grid | No accuracy gain (as Phase 3 predicted). Lasso found the 4 exact redundancies |
| 5. Logistic regression | "High demand" defined relative to each hour × day-type × year group, not one global threshold | A global threshold just means "is it rush hour?" (AUC 0.85 from the hour alone) |

## Tech Stack

Python · NumPy · pandas · scikit-learn · Matplotlib · Google Colab


## Dataset

UCI Bike Sharing dataset (hourly). Training data covers days 1–19 of each month; the test set is the 20th of every month.
