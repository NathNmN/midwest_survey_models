# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (pixi)
#     language: python
#     name: cours_ia_cyber_laval_exploration
# ---

# %% [markdown]
# # Compare machine learning models
#
# In this notebook, we will compare 3 pre-trained models that predict the
# **Census Region** of a respondent based on their survey answers.
#
# The 3 models are:
# - **Logistic Regression**: a simple linear model
# - **Random Forest**: a model based on many decision trees
# - **Gradient Boosting**: a model that builds trees sequentially

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# %%
# To simplify evaluation, we will group categories in the target to deal with a binary classification problem instead of a multiclass one.
y = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# %%
sample_idx = X.sample(n=1000, random_state=1).index
X_train = X.loc[sample_idx].reset_index(drop=True)
y_train = y.loc[sample_idx].reset_index(drop=True)
X_test = X.drop(sample_idx).reset_index(drop=True)
y_test = y.drop(sample_idx).reset_index(drop=True)

# %% [markdown]
# ## Load the 3 models
#
# The models were saved as `.pkl` files. We use `joblib` to load them.

# %%
import joblib
from midwest_survey_models.transformers import NumericalStabilizer

model_lr = joblib.load(r"C:\Users\natha\Downloads\cours_ia_cyber_laval_exploration-main2\cours_ia_cyber_laval_exploration-main/model_logistic_regression.pkl")
model_rf = joblib.load(r"C:\Users\natha\Downloads\cours_ia_cyber_laval_exploration-main2\cours_ia_cyber_laval_exploration-main/model_random_forest.pkl")
model_gb = joblib.load(r"C:\Users\natha\Downloads\cours_ia_cyber_laval_exploration-main2\cours_ia_cyber_laval_exploration-main/model_gradient_boosting.pkl")

# %% [markdown]
# Let's inspect what each model looks like. They are **pipelines**: they
# first transform the data, then make predictions.

# %%
model_lr

# %%
model_rf

# %%
model_gb

# %% [markdown]
# ## Evaluate the models with cross-validation
#
# To fairly evaluate each model, we use **cross-validation**.
# This means we train and test the model on different parts of the data multiple times, so we can see how well it generalizes.
#
# We use `cross_val_score` to get the score for every fold in cross-validation.

# %%
from sklearn.model_selection import cross_val_score

cv_lr = cross_val_score(model_lr, X, y, cv=5)
cv_rf = cross_val_score(model_rf, X, y, cv=5)
cv_gb = cross_val_score(model_gb, X, y, cv=5)

# %% [markdown]
# ## Question 6: Among the three models, which one has the best recall?
# Gradient Boosting a le meilleur recall.
# The **classification report** shows precision, recall, and f1-score for each class.
#
# - **Precision**: among all predictions for a class, how many were correct?
# - **Recall**: among all real examples of a class, how many were found?
# - **F1-score**: a balance between precision and recall
#
# We will define the positive class as "North Central".

# %%
y_pred_lr = model_lr.predict(X_test)

# %%
from skore import EstimatorReport
report = EstimatorReport(estimator = model_lr,
                X_test = X_test,
                y_test = y_test)
report.help()

# %%
report.metrics.summarize(pos_label="North Central").frame()
# %%
from sklearn.metrics import classification_report

for name, model in [("Logistic Regression", model_lr), ("Random Forest", model_rf), ("Gradient Boosting", model_gb)]:
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["North Central", "other"], zero_division=0))
# %% [markdown]
# Which model has the highest recall?

# %% [markdown]
# ## Question 7: Which model has the best practical application?
# Gradient Boosting est le plus rentable. La Régression Logistique est catastrophique car elle ne prédit jamais "North Central".
# Let's say that it costs 10 to make a false positive error, while it costs 1 to make a false negative error. Correctly predicting a positive example gains 5, while correctly predicting a negative example gains 2.
# %%
from sklearn.metrics import confusion_matrix

def profit_score(y_true, y_pred, pos_label="North Central"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[pos_label, "other"]).ravel()
    return tp * 5 + tn * 2 - fp * 10 - fn * 1

for name, model in [("Logistic Regression", model_lr), ("Random Forest", model_rf), ("Gradient Boosting", model_gb)]:
    y_pred = model.predict(X_test)
    score = profit_score(y_test, y_pred)
    print(f"{name}: profit = {score}")
# %%

# %% [markdown]
# Which model makes the most meaningful predictions in practice?

# %% [markdown]
# ## Question 8: Which model generalizes the best?
# La Régression Logistique a le plus petit gap mais c'est trompeur car elle prédit tout "other", donc elle ne sert à rien. Random Forest et Gradient Boosting overfittent tous les deux (train = 1.000), le Gradient Boosting # légèrement plus..
# To understand generalization, we compare the **training score** (how well the model fits the data it was trained on) with the **test score** (how well it performs on unseen data).
#
# A big gap between the two means the model is **overfitting**.  
#
# We don't want to do this only once, but several times. Use cross-validation for that. You can either use cross-validation from scikit-learn, or the CrossValidationReport from skore.
# %%
from sklearn.model_selection import cross_validate

for name, model in [("Logistic Regression", model_lr), ("Random Forest", model_rf), ("Gradient Boosting", model_gb)]:
    cv_results = cross_validate(model, X, y, cv=5, scoring="accuracy", return_train_score=True)
    train_mean = cv_results["train_score"].mean()
    test_mean  = cv_results["test_score"].mean()
    gap = train_mean - test_mean
    print(f"{name}: train={train_mean:.3f}, test={test_mean:.3f}, gap={gap:.3f}")
# %%

# %% [markdown]
# Which model has the smallest gap between train and test accuracy?
# That model generalizes the best.
#
# Which model has the largest gap? That model is likely **overfitting**.

# %%
# TODO: Based on the results above, which model would you choose
# for a real application? Write your answer as a comment below.

# My choice: Gradient Boosting
# Reason: Meilleur recall (0.99), meilleur profit (+5217), et bonne accuracy sur le test (0.924) malgré un léger overfitting.
# La Régression Logistique est inutilisable ici (recall = 0).

