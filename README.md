

# 📊 Cleaning and Preprocessing the Google Play Store Dataset: A Practical Guide with Pandas

**Subtitle:** A hands-on EDA walkthrough including data cleaning, missing value imputation, and feature engineering for robust analysis.

---

## 📌 Introduction

In data science, **exploratory data analysis (EDA)** is where the truth comes out. In this article, we dive deep into the Google Play Store dataset and walk through a structured approach to **data cleaning, transformation, and imputation**, all using Python's `pandas` library.

---

## 🧰 Step 1: Importing Required Libraries

```python
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## 📥 Step 2: Load the Dataset

```python
df = pd.read_csv('D:\\Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main\\Empty File\\googleplaystore.csv')
df.head()
```

Check the initial structure:

```python
df.shape
df.info()
df.describe()
```

Focus on numerical columns, especially `"Rating"`.

---

## 🔍 Step 3: Detecting and Removing Duplicates

```python
df[df.duplicated()]
df = df.drop_duplicates(keep='first')
df[df.duplicated()]
```

Duplicate removal **also eliminated some nulls**.

---

## 🚨 Step 4: Handling Missing Values

View nulls:

```python
df.isnull().sum()
```

The notorious **row 10472** had multiple missing fields and wrong formats — best removed:

```python
df = df.drop(index=10472)
```

Recheck:

```python
df.isnull().sum()
```

---

## 🔧 Step 5: Converting Data Types

Start with numerical-looking strings:

```python
df['Reviews'].str.isnumeric().sum()
```

Clean `Installs` and `Price` by removing symbols:

```python
cols_to_clean = ['Installs', 'Price']
df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(int)
df['Price'] = df['Price'].str.replace('$', '', regex=True).astype(float)
```

---

## 📆 Step 6: Parsing Dates

Convert `Last Updated` to `datetime`, extract components:

```python
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['Days'] = df['Last Updated'].dt.day
df['Month'] = df['Last Updated'].dt.month
df['Year'] = df['Last Updated'].dt.year
df.drop('Last Updated', axis=1, inplace=True)
```

---

## 🩹 Step 7: Imputing Missing Ratings

```python
df['Rating'].fillna(df['Rating'].median(), inplace=True)
```

**Why median?** It's robust to outliers and preserved original data variance better than mean imputation.

**Note:** Always compare **standard deviation and correlation matrices before and after** to validate imputation strategies.

---

## 🔄 Step 8: One-Hot Encoding Categorical Columns

Convert `Type` and `Content Rating`:

```python
df = pd.get_dummies(df, columns=['Type', 'Content Rating'], drop_first=True)
```

Also, `Type` had one missing — but its price was 0, so safely considered **Free**.

---

## 🧮 Step 9: Size Column — The Most Complex Fix

`Size` had values like "Varies with device". Initial attempts with mean/median produced poor results:

| Method | Std Dev | Variance | Drop in Variance |
|--------|---------|----------|------------------|
| Original | 23,997.81 | 575,894,944.03 | — |
| Mean Imputation | 22,159.30 | 491,034,410.56 | 84,860,533.47 |
| Median Imputation | 22,272.32 | 496,056,278.82 | 79,838,665.21 |
| **Category-wise Median** | **22,854.05** | **522,307,383.79** | **53,587,560.24** ✅ |

Thus, we used:

```python
df['Size'] = df.groupby('Category')['Size'].transform(lambda x: x.fillna(x.median()))
```

This preserved data integrity **better than any global method**.

---

## ✅ Final Note on Feature Engineering

While other columns offered more variables, adding them would **explode the feature space**, increasing chances of **overfitting** and **memory overload**. Hence, **no further feature engineering** was pursued.

---

## 📌 Conclusion

This article walked you through a **clean, reproducible** EDA and preprocessing pipeline for a real-world dataset. With a mix of best practices — from handling missing values to thoughtful imputation — we’ve prepared this dataset for robust machine learning downstream.

