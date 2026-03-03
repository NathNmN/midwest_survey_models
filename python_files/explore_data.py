# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Explore the Midwest Survey dataset
#
# In this notebook, we will explore the **Midwest Survey** dataset from [skrub](https://skrub-data.org/).
#
# This dataset contains survey responses from people across the United States,
# asking them about their perception of the Midwest region.
#
# The goal is to predict the **Census Region** where a respondent lives,
# based on their survey answers.

# %% [markdown]
# ## Load the dataset

# %%
from skrub.datasets import fetch_midwest_survey
import matplotlib.pyplot as plt
import pandas as pd

dataset = fetch_midwest_survey()

# X contains the features (the survey answers)
X = dataset.X
# y contains the target (the Census Region)
y = dataset.y

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?
# The dataset contains around 1600+ examples (respondents) and several dozen
# features. The exact number can be obtained using X.shape.
# The number of rows corresponds to the number of respondents.
# The number of columns corresponds to the number of survey questions used
# as predictive features.
# Use the `.shape` attribute to find out the number of rows and columns.

# %%
# Display the number of rows and columns
print("Shape of X:", X.shape)
print("Number of examples (rows):", X.shape[0])
print("Number of features (columns):", X.shape[1])

# %%
# You can also look at the first few rows of the dataset
X.head()

# %% [markdown]
# ## Question 2: What is the distribution of the target?
# After grouping the target into:
#   - "North Central"
#   - "other"
# The dataset is slightly imbalanced.
# The class "other" is generally more frequent than "North Central".
# However, the imbalance is moderate and not extreme.
# The target variable `y` tells us the Census Region of each respondent.
# Let's see how many respondents belong to each region.

# %%
# Count how many respondents belong to each region
target_counts = y.value_counts()
print(target_counts)

# %%
# Visualize the target distribution with a bar plot
target_counts.sort_values().plot(kind="barh")
plt.xlabel("Number of respondents")
plt.ylabel("Census Region")
plt.title("Distribution of Census Regions")
plt.show()

# %% [markdown]
# Is the target balanced (roughly the same number of examples per class) or imbalanced?

# %% [markdown]
# ## Question 3: What are the features that can be used to predict the target?
# All columns in X are used as predictive features.
# They include:
#   - Demographic information (Age, Gender, Education, Household_Income, etc.)
#   - Opinions about whether certain states belong to the Midwest
#   - Cultural perceptions
#   - Personal identification with the Midwest
# The dataset is mainly composed of categorical (text) variables,
# with only a few numerical variables.
# Let's look at the column names and their data types.

# %%
# List all column names
print("Column names:")
print(X.columns)

# %%
# Show data types for each column
print("\nData types:")
print(X.dtypes)

# %% [markdown]
# How many features are numerical? How many are categorical (text)?

# %%
num_features = X.select_dtypes(include=["number"]).shape[1]
cat_features = X.select_dtypes(exclude=["number"]).shape[1]

print("Number of numerical features:", num_features)
print("Number of categorical features:", cat_features)

# %%
from skrub import TableReport
TableReport(X)

# %% [markdown]
# ## Question 4: Are there any missing values in the dataset?
# Yes.
# There are NaN values in some columns.
# Additionally, some values such as:
#   - "Prefer not to answer"
#   - "Don't know"
# may represent implicit missing data.
# These should be considered carefully during preprocessing.
# Missing values can cause problems for machine learning models.
# Let's check if there are any.

# %%
# Check for NaN missing values
missing_counts = X.isna().sum()
print("Missing values per column:")
print(missing_counts[missing_counts > 0])

print("\nTotal missing values in dataset:", X.isna().sum().sum())

# %% [markdown]
# Missing values can sometimes be encoded differently. Let's look at some columns more closely.

# %%
# Look at unique values for the Household_Income column
print(X["Household_Income"].unique())

# %%
# Look at unique values for the Education column
print(X["Education"].unique())

# %% [markdown]
# Do you see a special value that could represent missing data?

# %% [markdown]
# ## Question 5: What is the most common answer to "How much do you personally identify as a Midwesterner"?
# "How much do you personally identify as a Midwesterner"?
# The most frequent answer is typically a moderate identification level,
# such as "Somewhat strongly" (or a similar intermediate category).
# This suggests that many respondents moderately identify as Midwesterners.
# Let's explore this important feature.

# %%
# display the value counts for the column
midwest_identity_counts = X[
    "How_much_do_you_personally_identify_as_a_Midwesterner"
].value_counts()

print(midwest_identity_counts)

# %%
# make a bar plot of the results
midwest_identity_counts.sort_values().plot(kind="barh")
plt.xlabel("Number of respondents")
plt.ylabel("Response")
plt.title("Identification as a Midwesterner")
plt.show()

# %% [markdown]
# ## Bonus: Explore another feature
#
# Pick another column and explore its distribution.
# For example: `Gender`, `Age`, or one of the
# "Do you consider X state as part of the Midwest" columns.

# %%
# Explore the Gender column
gender_counts = X["Gender"].value_counts()
print(gender_counts)

gender_counts.plot(kind="bar")
plt.xlabel("Gender")
plt.ylabel("Number of respondents")
plt.title("Gender Distribution")
plt.show()