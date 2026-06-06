# Descriptive Statistics and Summary Methods

## Lesson Overview

- This chapter covers descriptive statistics and summary methods in Pandas.
- Summarizing data distribution, central tendency, variation, and correlations is the first step in exploratory data analysis. Before modeling or engineering features, you must understand the scale, distributions, and patterns inside the variables.
- We will cover summary functions like `.describe()`, statistical aggregations (mean, median, standard deviation), correlation matrices using `.corr()`, value frequency counts using `.value_counts()`, and the behavior of calculations when missing values are present.
- Mastering these statistical tools allows you to profile datasets and extract insights.

## Learning Objectives

- Summarize numeric and categorical data distributions using the `.describe()` function.
- Compute central tendency (mean, median, mode) and dispersion (variance, standard deviation, quantiles) metrics.
- Track running logs using cumulative aggregations (`.cumsum()`, `.cummax()`, `.cummin()`).
- Analyze linear relationships between variables using correlation (`.corr()`) and covariance (`.cov()`) matrices.
- Extract unique values and calculate category frequency counts using `.value_counts()`.
- Configure axis alignments (`axis=0` vs `axis=1`) and missing data behaviors (`skipna`) across calculations.

---

## 1. Complete Dataset Summarization with `.describe()`

The `.describe()` method returns a summary of the distribution of variables in a DataFrame.

### Numeric Columns
Returns counts, mean, standard deviation, minimum, maximum, and quartile boundaries.

### Categorical / Object Columns
Returns counts, unique values count, the most frequent value (top), and its frequency.

### Setup for Demonstration

```python
import pandas as pd
import numpy as np

# Financial ledger log
df_ledger = pd.DataFrame({
    "Sector": ["Tech", "Retail", "Tech", "Energy", "Retail", "Tech"],
    "Revenue": [12000.50, 4500.00, np.nan, 21000.00, 5200.00, 14000.00],
    "Transactions": [120, 45, 110, 80, 55, np.nan],
    "Flagged": [False, False, True, False, False, False]
})

print("--- Master Ledger Table ---")
print(df_ledger)
```

### Output

```text
--- Master Ledger Table ---
   Sector   Revenue  Transactions  Flagged
0    Tech  12000.50         120.0    False
1  Retail   4500.00          45.0    False
2    Tech       NaN         110.0     True
3  Energy  21000.00          80.0    False
4  Retail   5200.00          55.0    False
5    Tech  14000.00           NaN    False
```

---

### Applying `.describe()`

```python
# Summarize numeric columns
print("--- Numeric Summary ---")
print(df_ledger.describe())

# Summarize categorical/object columns
print("\n--- Categorical Summary ---")
print(df_ledger["Sector"].describe())
```

### Output

```text
--- Numeric Summary ---
            Revenue  Transactions
count      5.000000      5.000000
mean   11340.100000     82.000000
std     6778.694635     33.279123
min     4500.000000     45.000000
25%     5200.000000     55.000000
50%    12000.500000     80.000000
75%    14000.000000    110.000000
max    21000.000000    120.000000

--- Categorical Summary ---
count        6
unique       3
top       Tech
freq         3
Name: Sector, dtype: object
```

---

## 2. Statistical Aggregations and Axis Controls

By default, statistical operations in Pandas calculate down the rows (**`axis=0`** or `axis='index'`) and ignore missing values (**`skipna=True`**).

To calculate across columns horizontally, specify **`axis=1`** or `axis='columns'`.

```python
# Calculate mean revenue down the rows (axis=0) ignoring NaNs
mean_rev = df_ledger["Revenue"].mean()
print("Mean Revenue (skipna=True):", mean_rev)

# Calculate mean revenue including NaNs (returns NaN if any entry is null)
mean_rev_nan = df_ledger["Revenue"].mean(skipna=False)
print("Mean Revenue (skipna=False):", mean_rev_nan)

# Compute mean across numeric columns for each row (axis=1)
row_means = df_ledger[["Revenue", "Transactions"]].mean(axis=1)
print("\n--- Horizontal Row Means ---")
print(row_means)
```

### Output

```text
Mean Revenue (skipna=True): 11340.1
Mean Revenue (skipna=False): nan

--- Horizontal Row Means ---
0     6060.25
1     2272.50
2      110.00
3    10540.00
4     2627.50
5    14000.00
dtype: float64
```

---

## 3. Cumulative Aggregations

Cumulative aggregations compute running summaries over the rows of a DataFrame:
- `.cumsum()`: Running total.
- `.cummax()`: Running maximum.
- `.cummin()`: Running minimum.
- `.cumprod()`: Running product.

```python
df_cum = pd.DataFrame({"Daily_Sales": [100, 150, 80, 200]})

# Compute cumulative running totals and peaks
df_cum["Running_Total"] = df_cum["Daily_Sales"].cumsum()
df_cum["Peak_Sales"] = df_cum["Daily_Sales"].cummax()

print("--- Cumulative Aggregations ---")
print(df_cum)
```

### Output

```text
--- Cumulative Aggregations ---
   Daily_Sales  Running_Total  Peak_Sales
0          100            100         100
1          150            250         150
2           80            330         150
3          200            530         200
```

---

## 4. Correlation and Covariance Matrices

Use `.corr()` to calculate the correlation matrix and `.cov()` to calculate the covariance matrix. By default, `.corr()` uses the Pearson correlation coefficient.

```python
# Stock prices and volumes DataFrame
df_market = pd.DataFrame({
    "Stock_Price": [150, 152, 149, 153, 155],
    "Volume_Millions": [12, 10, 15, 8, 7],
    "Interest_Rate": [5.2, 5.2, 5.3, 5.1, 5.0]
})

# Calculate correlation matrix
correlation_matrix = df_market.corr(method="pearson")

print("--- Market Correlation Matrix ---")
print(correlation_matrix)
```

### Output

```text
--- Market Correlation Matrix ---
                 Stock_Price  Volume_Millions  Interest_Rate
Stock_Price         1.000000        -0.908077      -0.916831
Volume_Millions    -0.908077         1.000000       0.957696
Interest_Rate      -0.916831         0.957696       1.000000
```
*Note: Stock_Price and Volume_Millions have a strong negative correlation (-0.90), indicating that as prices rise, volume tends to drop.*

---

## 5. Unique Values and Frequency Distributions

- `.unique()`: Returns an array of unique values in the Series.
- `.nunique()`: Returns the count of unique values.
- `.value_counts()`: Returns the frequency counts of unique values.

```python
# Find unique sectors
print("Unique Sectors:", df_ledger["Sector"].unique())
print("Number of Unique Sectors:", df_ledger["Sector"].nunique())

# Calculate frequency counts
print("\n--- Sector Frequency Counts ---")
print(df_ledger["Sector"].value_counts())

# Calculate frequency percentages (proportions)
print("\n--- Sector Proportions (%) ---")
print(df_ledger["Sector"].value_counts(normalize=True) * 100)
```

### Output

```text
Unique Sectors: ['Tech' 'Retail' 'Energy']
Number of Unique Sectors: 3

--- Sector Frequency Counts ---
Sector
Tech      3
Retail    2
Energy    1
Name: count, dtype: int64

--- Sector Proportions (%) ---
Sector
Tech      50.0
Retail    33.3
Energy    16.7
Name: proportion, dtype: float64
```

---

## Common Mistakes Students Make

- **Using `.corr()` on non-numeric columns**: Running `.corr()` on a DataFrame containing text columns will raise a `ValueError` in newer Pandas versions, or silently ignore string columns in older versions. Filter for numeric columns first: `df.select_dtypes(include=[np.number]).corr()`.
- **Misinterpreting `skipna=True`**: Students often assume that calculating the average of a column containing `NaN` values will result in a lower average. By default, `skipna=True` ignores missing values entirely, meaning the average is calculated using only the rows that contain valid data.
- **Applying `.unique()` to a DataFrame**: `.unique()` is a Series method; calling `df.unique()` raises an `AttributeError`. To find unique combinations across a DataFrame, use `df.drop_duplicates()`.
- **Confusing Correlation with Causation**: Finding a strong correlation coefficient (e.g. 0.95) does not prove that one variable causes the other. Both variables could be influenced by a third, unmeasured factor.

---

## Best Practices

- Run `df.describe()` immediately after loading a dataset to inspect value scales, ranges, and potential outliers.
- Filter DataFrames for numeric columns using `df.select_dtypes(include=[np.number])` before calculating correlation or covariance matrices.
- Use `df['Col'].value_counts(normalize=True)` to convert raw category counts into percentage distributions for reports.
- Set `skipna=False` if you want statistical calculations to flag the presence of missing values in the dataset.

---

## Worked Real-World Examples

### Worked Example 1: Classroom Performance Profiling

```python
import pandas as pd

# Classroom grades
grades = pd.DataFrame({
    "StudentID": [101, 102, 103, 104, 105],
    "Score": [85, 92, 78, 95, 88],
    "Attendance": [95, 98, 80, 99, 90]
})

# 1. Generate descriptive statistics summary
grade_summary = grades[["Score", "Attendance"]].describe()

# 2. Calculate correlation between Attendance and Exam Scores
attendance_score_corr = grades["Attendance"].corr(grades["Score"])

print("--- Grade Book Summary ---")
print(grade_summary)
print(f"\nCorrelation between Attendance and Score: {attendance_score_corr:.4f}")
```

### Output

```text
--- Grade Book Summary ---
           Score  Attendance
count   5.000000    5.000000
mean   86.200000   92.400000
std     7.049823    7.797435
min    78.000000   80.000000
25%    85.000000   90.000000
50%    88.000000   95.000000
75%    92.000000   98.000000
max    95.000000   99.000000

Correlation between Attendance and Score: 0.9632
```

---

### Worked Example 2: Financial Portfolio Peak Tracker

```python
import pandas as pd

# Daily stock returns
portfolio = pd.DataFrame({
    "Date": pd.date_range("2026-06-01", periods=5, freq="D"),
    "Returns": [1500, -800, 1200, -200, 3100]
})

# 1. Calculate the cumulative net balance over time
portfolio["Net_Balance"] = portfolio["Returns"].cumsum()

# 2. Track the cumulative peak balance achieved
portfolio["Peak_Balance"] = portfolio["Net_Balance"].cummax()

print("--- Portfolio Net Balance and Peaks ---")
print(portfolio)
```

### Output

```text
--- Portfolio Net Balance and Peaks ---
        Date  Returns  Net_Balance  Peak_Balance
0 2026-06-01     1500         1500          1500
1 2026-06-02     -800          700          1500
2 2026-06-03     1200         1900          1900
3 2026-06-04     -200         1700          1900
4 2026-06-05     3100         4800          4800
```

---

### Worked Example 3: Customer Satisfaction Distribution

```python
import pandas as pd

# Ingested customer survey reviews
reviews = pd.DataFrame({
    "ReviewID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Rating": ["Good", "Good", "Excellent", "Poor", "Good", "Excellent", "Poor", "Good"]
})

# Calculate the frequency counts and percentage distributions of ratings
ratings_dist = pd.DataFrame({
    "Counts": reviews["Rating"].value_counts(),
    "Proportion_Pct": reviews["Rating"].value_counts(normalize=True) * 100
})

print("--- Customer Ratings Distribution ---")
print(ratings_dist)
```

### Output

```text
--- Customer Ratings Distribution ---
           Counts  Proportion_Pct
Rating                           
Good            4            50.0
Excellent       2            25.0
Poor            2            25.0
```

---

## Practice Questions

1. Explain the differences in the outputs of `.describe()` when applied to a numeric column versus a categorical column.
2. Write a command to calculate the 90th percentile (quantile 0.90) of a Series `s`.
3. How does `skipna=True` (the default) handle missing values in statistical aggregations?
4. Write a command to calculate the Spearman rank correlation matrix for a DataFrame `df`.
5. Explain the differences between cumulative sum (`.cumsum()`) and moving window sum (`.rolling().sum()`).
6. Write a command to find the count of unique values in a Series `s`, excluding null values.
7. What is the value range of a Pearson correlation coefficient, and what do the extreme values signify?
8. Write a command to calculate the mean of each row in a DataFrame across all numeric columns.
9. How does `df.corr()` handle missing values (`NaN`) during calculation?
10. Describe how to extract the mode of a Series containing categorical string values.

---

## Mini Assignments

### Assignment 1: Corporate Salary Analytics
- Create an employee roster containing `Employee_ID`, `Department`, `Salary`, and `Years_Experience`.
- Generate descriptive statistical summaries for the numeric columns.
- Calculate the correlation coefficient between `Salary` and `Years_Experience`.

### Assignment 2: E-commerce Product Sales Cumulative Tracker
- Create a retail sales Series tracking daily revenues over 15 days.
- Calculate the cumulative sales revenue.
- Track the maximum daily sales revenue achieved up to each point in time using cumulative max.

### Assignment 3: CRM Customer Satisfaction Audit
- Load a client reviews dataset containing columns `CustomerID`, `Review_Rating` (strings: Excellent, Good, Fair, Poor), and `Recommend_Flag` (boolean).
- Calculate the frequency counts and percentage distributions of `Review_Rating`.
- Calculate the proportion of clients who recommend the product.

---

## Interview-Oriented Questions

- **What is the default correlation method used by `df.corr()`, and how does it differ from Spearman correlation?**
  - *Answer*: By default, `df.corr()` uses the Pearson correlation coefficient, which measures the linear relationship between continuous variables. Spearman correlation measures the monotonic relationship between variables using rank values, making it robust to outliers and suitable for non-linear relationships.
- **Explain the difference between variance (`.var()`) and standard deviation (`.std()`).**
  - *Answer*: Variance measures the average squared deviation of data points from the mean. Standard deviation is the square root of the variance, returning the dispersion metric back to the original unit scale of the data, which makes it easier to interpret.
- **How can we calculate descriptive statistics for both numeric and categorical columns in a single `.describe()` call?**
  - *Answer*: Pass the parameter `include='all'` to the `.describe()` method: `df.describe(include='all')`. This returns a combined summary table containing statistical metrics for all columns, filling irrelevant cells with `NaN`.
- **Explain how the `skipna` parameter affects aggregations and why you might set it to `False`.**
  - *Answer*: By default, `skipna=True` ignores missing values, calculating statistics using only the valid rows. Setting `skipna=False` forces the operation to return `NaN` if even a single missing value is present, helping developers identify and flag incomplete data.
- **What is the mathematical definition of a quartile, and how is it calculated in Pandas?**
  - *Answer*: Quartiles partition a sorted dataset into four equal parts. The 25th percentile (Q1) splits the bottom 25% of the data; the 50th percentile (Q2) represents the median; the 75th percentile (Q3) splits the top 25%. In Pandas, these are calculated using `.quantile([0.25, 0.5, 0.75])`.

---

## Teaching Notes for This Chapter

- **Deconstruct Correlation Visually**: Draw scatter plots showing positive correlation, negative correlation, and zero correlation to illustrate correlation values.
- **Contrast Cumulative vs Rolling**: Use a small table to illustrate the difference between running totals (`.cumsum()`) and moving averages (`.rolling()`).
- **Highlight Category Distributions**: Emphasize how `.value_counts(normalize=True)` simplifies calculating percentage distributions in reports.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `.describe()` to generate summary statistics for numeric and categorical columns.
- Aggregations default to calculating down the rows (`axis=0`) and ignoring missing values (`skipna=True`).
- Use `.corr()` to calculate correlation matrices to identify relationships between variables.
- Use `.cumsum()` and `.cummax()` to track running totals and peak values over time.
- Use `.value_counts(normalize=True)` to calculate percentage distributions for categorical columns.
- Pearson measures linear relationships, while Spearman measures monotonic rank relationships.
