# Pandas Series Fundamentals

## Lesson Overview

- This chapter teaches the Series object, which is one of the two most important data structures in Pandas.
- A Series is a one-dimensional labeled array.
- It stores data values and an index.
- You can think of it as a single column of data with row labels.
- This chapter builds the foundation for filtering, cleaning, transforming, and analyzing data in Pandas.

## Learning Objectives

- Understand what a Series is.
- Learn how a Series differs from Python lists, NumPy arrays, and DataFrame columns.
- Create Series from different data sources.
- Work with indexes and labels.
- Access, filter, and update values.
- Apply mathematical and logical operations.
- Handle missing values.
- Use useful built-in Series methods.
- Understand datatype behavior in Series.
- Learn how Series aligns data automatically by index.
- Use Series in real-world examples.

## What is a Pandas Series?

- A Pandas Series is a one-dimensional data structure.
- It contains:
- Data values.
- Index labels.
- Optional name.
- A Series can hold integers, floats, strings, booleans, dates, and many other types.
- Each element has a label, which makes Series more powerful than a plain Python list.

### Basic idea

- A Python list stores only values.
- A Series stores values with labels.
- Those labels are called the index.

### Example

```python
import pandas as pd

marks = pd.Series([85, 90, 78, 92])
print(marks)
```

### Output

```python
0    85
1    90
2    78
3    92
dtype: int64
```

- The left side is the index.
- The right side is the value.
- By default, Pandas creates integer indexes starting from 0.

## Why Series is Important

- It is the base building block of a DataFrame.
- Many Pandas operations start by understanding Series.
- Filtering columns from a DataFrame often returns a Series.
- Series supports labels, vectorized operations, missing values, and index alignment.

## Real-Life Examples of Series

- Monthly sales of a shop.
- Temperatures across seven days.
- Student marks in one subject.
- Product prices by product name.
- Population by city.
- Website visits by date.

### Example: Product prices

```python
prices = pd.Series([499, 899, 1299], index=["Mouse", "Keyboard", "Monitor"])
print(prices)
```

### Output

```python
Mouse        499
Keyboard     899
Monitor     1299
dtype: int64
```

## Series vs List vs NumPy Array vs DataFrame

| Structure | Dimension | Labels | Best Use |
|----------|-----------|--------|----------|
| Python List | 1D | No | General-purpose storage |
| NumPy Array | 1D or more | No | Fast numerical operations |
| Pandas Series | 1D | Yes | Labeled single-column data |
| Pandas DataFrame | 2D | Yes | Table-like data analysis |

### Main difference

- A Series has labels.
- A Series can align values based on those labels.
- A DataFrame is made of multiple Series.

## Anatomy of a Series

A Series mainly has these parts:

- Values.
- Index.
- Dtype.
- Name.
- Size.
- Shape.

### Example

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="numbers")

print("Values:")
print(s.values)

print("Index:")
print(s.index)

print("Data type:")
print(s.dtype)

print("Name:")
print(s.name)

print("Size:")
print(s.size)

print("Shape:")
print(s.shape)
```

## Creating a Series

## 1. Create Series from a List

```python
import pandas as pd

numbers = pd.Series([10, 20, 30, 40])
print(numbers)
```

## 2. Create Series with Custom Index

```python
numbers = pd.Series([10, 20, 30], index=["x", "y", "z"])
print(numbers)
```

## 3. Create Series from a Dictionary

- Dictionary keys become index labels.
- Dictionary values become Series values.

```python
student_marks = pd.Series({
    "Math": 88,
    "Science": 91,
    "English": 84
})
print(student_marks)
```

## 4. Create Series from a Scalar Value

- A scalar means one single value.
- You must provide an index if you want repeated values.

```python
status = pd.Series("Present", index=["Amit", "Riya", "Karan"])
print(status)
```

### Output

```python
Amit      Present
Riya      Present
Karan     Present
dtype: object
```

## 5. Create Series from a NumPy Array

```python
import numpy as np
import pandas as pd

arr = np.array([5, 10, 15, 20])
s = pd.Series(arr)
print(s)
```

## 6. Create Empty Series

```python
empty_series = pd.Series(dtype="float64")
print(empty_series)
```

## Understanding the Index

- Every Series has an index.
- Index gives labels to each row.
- Index can be numbers, strings, dates, or custom identifiers.

### Example with custom labels

```python
cities = pd.Series([22000000, 19000000, 13000000], index=["Delhi", "Mumbai", "Bengaluru"])
print(cities)
```

### Why index matters

- It makes data readable.
- It allows label-based access.
- It helps automatic alignment during operations.

## Accessing Values in a Series

## 1. Access by Position

```python
s = pd.Series([100, 200, 300, 400])
print(s[0])
print(s [pandas.pydata](https://pandas.pydata.org/docs/))
```

## 2. Access by Label

```python
s = pd.Series([100, 200, 300], index=["a", "b", "c"])
print(s["a"])
print(s["c"])
```

## 3. Access Multiple Values

```python
s = pd.Series([100, 200, 300], index=["a", "b", "c"])
print(s[["a", "c"]])
```

## 4. Slicing

### Position-based slicing

```python
s = pd.Series([10, 20, 30, 40, 50])
print(s[1:4])
```

### Label-based slicing

```python
s = pd.Series([10, 20, 30, 40, 50], index=["a", "b", "c", "d", "e"])
print(s["b":"d"])
```

- In label slicing, the ending label is usually included.

## loc and iloc with Series

- `loc` is label-based.
- `iloc` is position-based.

### Example

```python
s = pd.Series([500, 600, 700], index=["x", "y", "z"])

print(s.loc["y"])
print(s.iloc [pandas.pydata](https://pandas.pydata.org/docs/user_guide/index.html))
```

### Multiple selection

```python
print(s.loc[["x", "z"]])
print(s.iloc[[0, 2]])
```

## Updating Values in a Series

## 1. Update a single value

```python
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
s["b"] = 200
print(s)
```

## 2. Update multiple values

```python
s[["a", "c"]] = [100, 300]
print(s)
```

## 3. Update using condition

```python
s = pd.Series([5, 15, 25, 35])
s[s > 20] = 999
print(s)
```

## Adding New Elements

- You can add a new label and assign a value.

```python
s = pd.Series([10, 20], index=["a", "b"])
s["c"] = 30
print(s)
```

## Deleting Elements

## Using `drop()`

```python
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
s = s.drop("b")
print(s)
```

- `drop()` returns a new Series unless reassigned.

## Series Attributes

Some important Series attributes are:

- `index`
- `values`
- `dtype`
- `shape`
- `size`
- `name`
- `ndim`
- `empty`

### Example

```python
s = pd.Series([12, 24, 36], name="multiples")

print(s.index)
print(s.values)
print(s.dtype)
print(s.shape)
print(s.size)
print(s.name)
print(s.ndim)
print(s.empty)
```

## Naming a Series

- A Series can have a name.
- Naming is useful when converting to DataFrame or for reporting.

```python
sales = pd.Series([1000, 1200, 900], index=["Jan", "Feb", "Mar"], name="Monthly Sales")
print(sales)
print(sales.name)
```

## Changing Index Labels

```python
s = pd.Series([1, 2, 3], index=["a", "b", "c"])
s.index = ["x", "y", "z"]
print(s)
```

## Boolean Filtering

- Boolean filtering is one of the most useful concepts in Pandas.
- It allows selecting values based on conditions.

### Example

```python
marks = pd.Series([45, 78, 88, 33, 91])

print(marks > 50)
print(marks[marks > 50])
```

### Real example: passing students

```python
student_marks = pd.Series([35, 67, 89, 42, 90], index=["A", "B", "C", "D", "E"])
passed = student_marks[student_marks >= 40]
print(passed)
```

## Conditional Replacement

```python
ages = pd.Series([12, 17, 25, 14, 30])
ages[ages < 18] = "Minor"
print(ages)
```

- Be careful with mixed data types.
- Replacing integers with strings changes the dtype to object.

## Arithmetic Operations on Series

- Series supports arithmetic operations like addition, subtraction, multiplication, division, and more.
- These operations are vectorized, so they work on all values at once.

### Example

```python
s = pd.Series([10, 20, 30])
print(s + 5)
print(s * 2)
print(s / 10)
```

## Arithmetic between two Series

```python
s1 = pd.Series([10, 20, 30], index=["a", "b", "c"])
s2 = pd.Series([1, 2, 3], index=["a", "b", "c"])

print(s1 + s2)
print(s1 - s2)
```

## Index Alignment in Series

- This is one of the most important concepts in Pandas.
- When two Series are combined, Pandas aligns them by index label, not by position.

### Example

```python
s1 = pd.Series([100, 200, 300], index=["a", "b", "c"])
s2 = pd.Series([10, 20, 30], index=["b", "c", "d"])

print(s1 + s2)
```

### Output explanation

- `a` has no matching value in `s2`, so result becomes `NaN`.
- `b` matches `b`.
- `c` matches `c`.
- `d` has no matching value in `s1`, so result becomes `NaN`.

### Safer alternative with fill value

```python
result = s1.add(s2, fill_value=0)
print(result)
```

## Comparison Operations

```python
s = pd.Series([10, 20, 30, 40])

print(s > 20)
print(s == 30)
print(s != 10)
print(s <= 25)
```

## Logical Filtering with Multiple Conditions

```python
s = pd.Series([12, 25, 37, 48, 59, 61])

filtered = s[(s > 20) & (s < 60)]
print(filtered)
```

### Important note

- Use `&` for AND.
- Use `|` for OR.
- Use parentheses around each condition.

## Missing Values in Series

- Missing values are very common in real data.
- Pandas usually represents missing values as `NaN`.
- Some newer data types may also use `pd.NA`.

## Creating Series with missing values

```python
import pandas as pd
import numpy as np

s = pd.Series([10, np.nan, 30, None, 50])
print(s)
```

## Detecting missing values

```python
print(s.isna())
print(s.isnull())
```

- `isna()` and `isnull()` do the same job.

## Detecting non-missing values

```python
print(s.notna())
print(s.notnull())
```

## Removing missing values

```python
cleaned = s.dropna()
print(cleaned)
```

## Filling missing values

```python
filled = s.fillna(0)
print(filled)
```

### Fill with mean

```python
s = pd.Series([10, 20, None, 40, 50])
filled = s.fillna(s.mean())
print(filled)
```

## Series Data Types

- Every Series has a dtype.
- Dtype tells what kind of data is stored.

### Common dtypes

- `int64`
- `float64`
- `object`
- `bool`
- `datetime64[ns]`
- `category`
- `string`

### Example

```python
s1 = pd.Series([1, 2, 3])
s2 = pd.Series([1.5, 2.5, 3.5])
s3 = pd.Series(["A", "B", "C"])
s4 = pd.Series([True, False, True])

print(s1.dtype)
print(s2.dtype)
print(s3.dtype)
print(s4.dtype)
```

## Type Conversion in Series

### Convert to float

```python
s = pd.Series([1, 2, 3])
print(s.astype(float))
```

### Convert string numbers to int

```python
s = pd.Series(["10", "20", "30"])
print(s.astype(int))
```

### Convert safely with numeric parsing

```python
s = pd.Series(["10", "20", "abc", "40"])
converted = pd.to_numeric(s, errors="coerce")
print(converted)
```

- Invalid values become `NaN` when `errors="coerce"` is used.

## Series Methods for Inspection

## `head()`

```python
s = pd.Series(range(1, 11))
print(s.head())
```

## `tail()`

```python
print(s.tail())
```

## `sample()`

```python
print(s.sample(3))
```

## `describe()`

```python
nums = pd.Series([10, 20, 30, 40, 50])
print(nums.describe())
```

## `info()`

- `info()` is more common with DataFrame.
- It is less commonly used with Series, but still available in many cases through object inspection patterns.
- Usually for Series, `dtype`, `shape`, `count()`, and `isna().sum()` are more practical.

## Summary Statistics on Series

### Numeric Series methods

```python
s = pd.Series([10, 20, 30, 40, 50])

print(s.sum())
print(s.mean())
print(s.median())
print(s.min())
print(s.max())
print(s.std())
print(s.var())
print(s.count())
```

### Example: business sales

```python
sales = pd.Series([2500, 3000, 2800, 3500, 4000], index=["Mon", "Tue", "Wed", "Thu", "Fri"])

print("Total:", sales.sum())
print("Average:", sales.mean())
print("Highest:", sales.max())
print("Lowest:", sales.min())
```

## Value Counts and Unique Values

## `unique()`

```python
colors = pd.Series(["red", "blue", "red", "green", "blue"])
print(colors.unique())
```

## `nunique()`

```python
print(colors.nunique())
```

## `value_counts()`

```python
print(colors.value_counts())
```

### Real example: survey responses

```python
responses = pd.Series(["Yes", "No", "Yes", "Yes", "No", "Maybe", "Yes"])
print(responses.value_counts())
```

## Sorting a Series

## Sort by values

```python
s = pd.Series([30, 10, 50, 20], index=["a", "b", "c", "d"])
print(s.sort_values())
```

## Sort by index

```python
print(s.sort_index())
```

## Ranking Values

```python
scores = pd.Series([85, 95, 70, 95, 60])
print(scores.rank())
print(scores.rank(ascending=False))
```

## Checking Membership

```python
s = pd.Series([100, 200, 300], index=["a", "b", "c"])

print("a" in s.index)
print(200 in s.values)
```

## String Operations on Series

- If a Series contains strings, you can use `.str`.

### Example

```python
names = pd.Series(["amit", "riya", "karan", "neha"])

print(names.str.upper())
print(names.str.title())
print(names.str.len())
```

### Filtering string patterns

```python
emails = pd.Series(["a@gmail.com", "b@yahoo.com", "c@gmail.com"])
print(emails[emails.str.contains("gmail")])
```

## Datetime Operations on Series

- If a Series contains dates, you can use `.dt`.

### Example

```python
dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-02-15", "2026-03-20"]))

print(dates.dt.year)
print(dates.dt.month)
print(dates.dt.day)
```

## Categorical Data in Series

- Categories help save memory and improve performance for repeated text values.

```python
s = pd.Series(["High", "Low", "Medium", "High", "Low"], dtype="category")
print(s)
print(s.dtype)
```

## Applying Functions to Series

## Using `apply()`

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s.apply(lambda x: x * 10))
```

## Using built-in functions

```python
print(s.apply(abs))
```

### Example: grade labels

```python
marks = pd.Series([45, 67, 82, 91, 38])

grades = marks.apply(lambda x: "Pass" if x >= 40 else "Fail")
print(grades)
```

## Mapping Values in Series

- `map()` is useful when converting one set of values into another.

```python
days = pd.Series(["Mon", "Tue", "Wed", "Thu"])
mapped = days.map({
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday"
})
print(mapped)
```

### Example: convert codes to names

```python
dept_codes = pd.Series(["HR", "IT", "FIN", "IT", "HR"])
dept_names = dept_codes.map({
    "HR": "Human Resources",
    "IT": "Information Technology",
    "FIN": "Finance"
})
print(dept_names)
```

## Replacing Values

```python
s = pd.Series(["Poor", "Average", "Good", "Poor"])
print(s.replace("Poor", "Low"))
```

### Replace multiple values

```python
s = pd.Series(["A", "B", "C", "A"])
print(s.replace({"A": "Excellent", "B": "Good", "C": "Average"}))
```

## Checking Duplicates

```python
s = pd.Series(["apple", "banana", "apple", "orange", "banana"])

print(s.duplicated())
print(s.drop_duplicates())
```

## Clipping Values

- Clipping limits values to a minimum or maximum range.

```python
s = pd.Series([5, 10, 15, 20, 25])
print(s.clip(lower=10, upper=20))
```

## Combining Series

## `combine_first()`

- This is useful when one Series has missing values and another can fill them.

```python
s1 = pd.Series([10, None, 30], index=["a", "b", "c"])
s2 = pd.Series([100, 200, 300], index=["a", "b", "c"])

result = s1.combine_first(s2)
print(result)
```

## `append()` note

- Older tutorials often mention `append()`.
- In modern Pandas, using `pd.concat()` is the better approach.

### Example

```python
s1 = pd.Series([1, 2, 3])
s2 = pd.Series([4, 5])

combined = pd.concat([s1, s2], ignore_index=True)
print(combined)
```

## Converting Series to Other Formats

## Convert to list

```python
s = pd.Series([10, 20, 30])
print(s.to_list())
```

## Convert to dictionary

```python
s = pd.Series([100, 200], index=["A", "B"])
print(s.to_dict())
```

## Convert to DataFrame

```python
s = pd.Series([88, 92, 76], index=["Math", "Science", "English"], name="Marks")
df = s.to_frame()
print(df)
```

## Reset Index

- Resetting index converts the index into a regular column when needed.

```python
s = pd.Series([100, 200, 300], index=["a", "b", "c"], name="values")
print(s.reset_index())
```

## Useful Index-Based Operations

## Reindexing

```python
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s.reindex(["a", "b", "c", "d"]))
```

### Reindex with fill value

```python
print(s.reindex(["a", "b", "c", "d"], fill_value=0))
```

## Rename index labels

```python
s = pd.Series([10, 20, 30], index=["jan", "feb", "mar"])
print(s.rename(index={"jan": "January", "feb": "February"}))
```

## Element-wise Math Functions

```python
import numpy as np
import pandas as pd

s = pd.Series([1, 4, 9, 16])

print(np.sqrt(s))
print(np.log(s))
```

## Comparison with Scalar and Another Series

```python
s1 = pd.Series([10, 20, 30])
s2 = pd.Series([10, 25, 30])

print(s1 == 20)
print(s1 == s2)
print(s1 > s2)
```

## Memory and Performance Notes

- Series operations are vectorized, so they are usually faster than Python loops.
- For large datasets, avoid unnecessary loops over Series.
- Prefer built-in methods and direct Pandas operations.
- Use appropriate dtypes such as category for repeated text values.
- Newer Pandas versions continue improving datatype handling and consistency.

## Common Mistakes Students Make

- Confusing index labels with positions.
- Using plain brackets without understanding whether access is label-based or position-based.
- Forgetting that Series operations align by index.
- Mixing strings and numbers in the same Series without realizing dtype changes.
- Forgetting to handle missing values before analysis.
- Using loops where vectorized operations are better.
- Assuming two Series add by order instead of by label.
- Misusing `&` and `|` without parentheses.

## Best Practices

- Give meaningful names to Series.
- Use custom indexes when labels add meaning.
- Prefer `loc` and `iloc` for clarity.
- Always inspect `dtype`.
- Check for missing values before analysis.
- Use vectorized operations instead of loops.
- Use `value_counts()` for quick category analysis.
- Use `map()` and `replace()` carefully depending on the task.
- Use `pd.concat()` instead of older append-style patterns.

## Worked Example 1: Student Marks

```python
import pandas as pd

marks = pd.Series([78, 85, 92, 67, 88], index=["Amit", "Riya", "Karan", "Neha", "Vikas"], name="Math Marks")

print(marks)
print("Average:", marks.mean())
print("Highest:", marks.max())
print("Passed students:")
print(marks[marks >= 40])
```

## Worked Example 2: Monthly Sales

```python
sales = pd.Series([25000, 27000, 30000, 28000, 35000], index=["Jan", "Feb", "Mar", "Apr", "May"], name="Sales")

print(sales)
print("Total Sales:", sales.sum())
print("Months above 28000:")
print(sales[sales > 28000])
print("Sorted Sales:")
print(sales.sort_values(ascending=False))
```

## Worked Example 3: Missing Attendance Data

```python
attendance = pd.Series([90, None, 85, 88, None], index=["Amit", "Riya", "Karan", "Neha", "Vikas"])

print("Original:")
print(attendance)

print("Missing values:")
print(attendance.isna())

print("Filled with average:")
print(attendance.fillna(attendance.mean()))
```

## Worked Example 4: Product Price Update

```python
prices = pd.Series([499, 899, 1299], index=["Mouse", "Keyboard", "Monitor"])

prices["Keyboard"] = 999
prices["Speaker"] = 1599

print(prices)
print("Expensive products:")
print(prices[prices > 900])
```

## Worked Example 5: Department Mapping

```python
employees = pd.Series(["HR", "IT", "IT", "FIN", "HR", "MKT"])
department_names = employees.map({
    "HR": "Human Resources",
    "IT": "Information Technology",
    "FIN": "Finance",
    "MKT": "Marketing"
})

print(department_names)
```

## Worked Example 6: Index Alignment

```python
store1 = pd.Series([100, 200, 300], index=["Pen", "Pencil", "Eraser"])
store2 = pd.Series([10, 20, 30], index=["Pencil", "Eraser", "Scale"])

print(store1 + store2)
print(store1.add(store2, fill_value=0))
```

## Practice Questions

- Create a Series of five student names with their marks as values.
- Create a Series from a dictionary of product prices.
- Access the second and fourth values of a Series.
- Filter values greater than 50 from a numeric Series.
- Replace all values below 40 with `"Fail"`.
- Find the mean, median, and standard deviation of a Series.
- Count unique values in a Series of cities.
- Create two Series with partially matching indexes and add them.
- Create a Series with missing values and fill them with the mean.
- Convert a Series into a DataFrame.

## Mini Assignments

## Assignment 1: Exam Result Analysis

- Create a Series of marks for 10 students.
- Find top scorer.
- Find average marks.
- Filter students who scored above average.
- Convert marks into grade labels.

## Assignment 2: Shop Sales Tracker

- Create a Series of daily sales for 7 days.
- Find total weekly sales.
- Find highest and lowest sales day.
- Apply a 10 percent increase to all sales.
- Sort the results in descending order.

## Assignment 3: Employee Data Cleaning

- Create a Series of employee departments with repeated values.
- Count frequency of each department.
- Replace abbreviations with full names.
- Convert department column into category dtype.

## Interview-Oriented Questions

- What is a Pandas Series?
- How is a Series different from a list?
- How is a Series different from a DataFrame column?
- What is index alignment in Series?
- What happens when two Series have different indexes?
- How do `loc` and `iloc` differ?
- How do you handle missing values in a Series?
- What is the difference between `map()` and `replace()`?
- Why are vectorized operations preferred over loops?
- When would you use category dtype in a Series?

## Teaching Notes for This Chapter

- Start with the idea of a labeled column.
- Compare Series with Excel columns and SQL result columns.
- Spend extra time on index alignment because students usually miss it.
- Show both default index and custom index examples.
- Use business, student, and ecommerce examples.
- Teach missing values early because real datasets always contain them.
- Include many small code snippets before moving to larger examples.

## Chapter Wrap-up Concepts Students Must Master

- A Series is one-dimensional and labeled.
- It contains values and an index.
- It can be created from lists, dictionaries, arrays, and scalars.
- Values can be accessed by position and label.
- Boolean filtering is essential.
- Missing values must be handled properly.
- Arithmetic works element-wise and aligns by index.
- Series supports rich methods for summary, cleaning, transformation, and analysis.
- Series is the foundation of DataFrames and many real-world Pandas workflows.