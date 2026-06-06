# Options, Settings, and Display Customization

## Lesson Overview

- This chapter covers Pandas' global configuration system. We explore modifying output display dimensions, customizing decimal format layouts, and managing settings temporarily using context managers.
- When inspecting large DataFrames in Jupyter or the terminal, Pandas defaults often truncate columns or print floats with excessive decimal places. This makes visual inspections difficult and disrupts data reporting.
- We will cover methods like `pd.set_option()`, `pd.get_option()`, `pd.reset_option()`, and use the `pd.option_context()` context manager.
- Mastering these configuration tools allows you to customize how your tables are displayed in reports and notebooks.

## Learning Objectives

- Retrieve and modify global options using `pd.get_option()` and `pd.set_option()`.
- Customize output dimensions by setting limits on rows, columns, and characters.
- Format decimal numbers globally using the `display.float_format` configuration.
- Apply temporary display settings within code blocks using `pd.option_context()`.
- Restore configurations back to default settings using `pd.reset_option()`.

---

## The Pandas Option Interface

Pandas has a global options system that controls display behavior, warning levels, and computational limits.
You can configure options using:
- **`pd.set_option(option_name, value)`**: Set a global option value.
- **`pd.get_option(option_name)`**: Retrieve the current option value.
- **`pd.reset_option(option_name)`**: Restore the option back to its default value.

### Common Options
- **`display.max_rows`**: Maximum number of rows to display before truncating.
- **`display.max_columns`**: Maximum number of columns to display.
- **`display.float_format`**: Custom formatting for floating-point values.
- **`display.max_colwidth`**: Maximum character width for cells before truncating with `...`.

---

### 1. Adjusting Display Dimensions

By default, Pandas truncates large DataFrames to keep outputs compact. You can increase these limits to inspect complete tables.

```python
import pandas as pd

# Show the default values for rows and columns display options
default_rows = pd.get_option("display.max_rows")
default_cols = pd.get_option("display.max_columns")
print(f"Default display limits: {default_rows} rows, {default_cols} columns")

# Set display limits to show up to 100 rows and 50 columns
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 50)

# Reset options back to their defaults
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
```

### Output

```text
Default display limits: 60 rows, 20 columns
```

---

## 2. Formatting Floating-Point Numbers Globally

You can define a custom float formatting function to suppress scientific notation or limit decimal places globally.

```python
import numpy as np

# Sample DataFrame containing large values and multiple decimals
df_money = pd.DataFrame({
    "Gross_Revenue": [12500000.456, 850000.123],
    "Tax_Rate": [0.18345, 0.05212]
})

print("--- Default Float Display ---")
print(df_money)

# Set global float formatter to display two decimal places
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print("\n--- Customized Float Display ---")
print(df_money)

# Reset formatting back to default
pd.reset_option("display.float_format")
```

### Output

```text
--- Default Float Display ---
   Gross_Revenue  Tax_Rate
0   1.250000e+07   0.18345
1   8.500001e+05   0.05212

--- Customized Float Display ---
   Gross_Revenue  Tax_Rate
0  12,500,000.46      0.18
1     850,000.12      0.05
```

---

## 3. Temporary Configurations with `pd.option_context()`

Modifying options globally can affect other parts of your code. To apply display settings temporarily, use the **`pd.option_context()`** context manager inside a `with` block. Settings revert to their previous values when the block exits.

```python
df_large = pd.DataFrame(np.random.rand(10, 10))

# Temporarily display all rows and columns within the with block
with pd.option_context("display.max_rows", 100, "display.max_columns", 50):
    print("--- Inside Context Block ---")
    print(df_large.head(2))

# Settings are automatically reverted outside the block
```

### Output

```text
--- Inside Context Block ---
          0    1    2    3    4    5    6    7    8    9
0  0.5488135 0.71 0.60 0.54 0.42 0.64 0.43 0.89 0.96 0.38
1  0.7917250 0.52 0.56 0.92 0.07 0.08 0.02 0.83 0.77 0.87
```

---

## Common Mistakes Students Make

- **Leaving global settings modified**: Setting `pd.set_option('display.max_rows', None)` displays the entire DataFrame. If you print a DataFrame with 1,000,000 rows, this can crash your Jupyter Notebook or terminal. Always use `pd.option_context()` for temporary large displays.
- **Typing incorrect option names**: Passing a misspelled option name (e.g. `pd.set_option('max_rows', 10)`) raises a `NoSuchOptionError`. Ensure the prefix is included (e.g. `'display.max_rows'`).
- **Assuming option changes modify the data**: Setting float formats (like `display.float_format`) only changes how numbers are displayed in the console or notebook; it does not round or modify the underlying numeric values in the DataFrame.

---

## Best Practices

- Use `pd.option_context()` to apply display settings temporarily rather than modifying options globally.
- Set a float formatter globally (`pd.set_option('display.float_format', ...)`) in financial scripts to suppress scientific notation and standardize decimal displays.
- Reset modified configurations using `pd.reset_option('all')` at the beginning of clean notebooks.
- Set `display.max_colwidth` to a higher value when inspecting columns that contain long strings (like URLs or customer comments) to prevent text truncation.

---

## Worked Real-World Examples

### Worked Example 1: Formatting Financial Statements for Meetings

```python
import pandas as pd

# Revenue ledger
df_ledger = pd.DataFrame({
    "Product": ["Enterprise Software", "Standard Support License"],
    "US_Revenue": [12500500.75, 450000.20],
    "Tax_Multiplier": [1.1852, 1.0500]
})

# Display with meeting-ready settings (currency formatting)
with pd.option_context("display.float_format", lambda x: f"${x:,.2f}" if x > 100 else f"{x:.4f}"):
    print("--- Corporate Revenue Report ---")
    print(df_ledger)
```

### Output

```text
--- Corporate Revenue Report ---
                    Product     US_Revenue  Tax_Multiplier
0       Enterprise Software $12,500,500.75          1.1852
1  Standard Support License    $450,000.20          1.0500
```

---

### Worked Example 2: Inspecting Raw Text Logs

```python
import pandas as pd

# System events list
df_events = pd.DataFrame({
    "EventID": [101, 102],
    "Log_Message": [
        "ERROR: Database connection timeout on node A during hourly batch write run.",
        "WARN: API response latency exceeds 500ms on secondary region validation checker."
    ]
})

# Display logs without character truncation
with pd.option_context("display.max_colwidth", None):
    print("--- Unclipped System Logs ---")
    print(df_events)
```

### Output

```text
--- Unclipped System Logs ---
   EventID                                                                        Log_Message
0      101       ERROR: Database connection timeout on node A during hourly batch write run.
1      102  WARN: API response latency exceeds 500ms on secondary region validation checker.
```

---

## Practice Questions

1. Explain the differences between the `pd.set_option()`, `pd.get_option()`, and `pd.reset_option()` methods in Pandas.
2. Write a command to set the maximum column width display limit to 150 characters.
3. Why is it recommended to use `pd.option_context()` instead of modifying options globally?
4. Write a command to format all floats in a DataFrame to display as percentages (e.g. `85.5%`).
5. What error occurs when you pass an invalid option name to `pd.set_option()`?
6. Write a command to reset all modified options back to their defaults at once.
7. Explain how the `display.precision` option differs from the `display.float_format` option.
8. Write a script that temporarily increases the displayed column count to 100 within a single cell.
9. How can you display all columns of a DataFrame without truncation?
10. Describe how to suppress scientific notation globally for floating-point numbers in Pandas.

---

## Mini Assignments

### Assignment 1: Corporate Sales Currency Display
- Create a corporate sales DataFrame with columns `Revenue` (large float numbers) and `Tax_Rate` (four decimal places).
- Set the display format globally to show `Revenue` with commas and two decimal places, and `Tax_Rate` as a percentage.
- Revert the settings back to their defaults post-inspection.

### Assignment 2: Untruncated Text Log Audit
- Create a DataFrame containing 5 log entries with long text messages.
- Use `pd.option_context()` to print the DataFrame without column truncation, ensuring the full text of the log messages is visible.

### Assignment 3: Resetting All Custom Options
- Modify multiple Pandas options, including `display.max_rows`, `display.max_columns`, and `display.precision`.
- Verify the updated settings, and then reset them all to default settings in a single command.

---

## Interview-Oriented Questions

- **What is the purpose of the `pd.option_context()` method, and what are its core advantages?**
  - *Answer*: `pd.option_context()` is a context manager used to apply display settings temporarily within a `with` block. Its main advantage is that it prevents display settings from leaking and affecting other parts of your code, automatically restoring the previous configurations when the block exits.
- **Does modifying display options like `display.float_format` change the values stored in the DataFrame?**
  - *Answer*: No, display options only change how the data is rendered in the console or Jupyter Notebook. The underlying values and data types stored in the DataFrame memory remain unmodified, preserving precision for calculations.
- **How can we configure Pandas to show all columns of a DataFrame without truncation?**
  - *Answer*: Set the `display.max_columns` option to `None`: `pd.set_option('display.max_columns', None)`. This tells the display engine to print all columns in the DataFrame.
- **Explain the difference between `display.max_rows` and `display.min_rows`.**
  - *Answer*: `display.max_rows` defines the maximum number of rows to display before truncating. If a DataFrame has more rows than `max_rows`, Pandas truncates the output and displays only the number of rows specified by `display.min_rows` (split between the top and bottom of the table).
- **What is a `NoSuchOptionError`, and how can we prevent it?**
  - *Answer*: A `NoSuchOptionError` is raised when you pass a misspelled or invalid option name to a configuration method (e.g. `pd.set_option('max_rows', 10)`). To prevent it, verify the option name against the official documentation and ensure the namespace prefix (e.g. `'display.'`) is included.

---

## Teaching Notes for This Chapter

- **Demonstrate temporary vs global changes**: Show the difference between setting options globally and using a context manager in class.
- **Highlight notebook crashes**: Warn students about the risks of displaying massive datasets (setting `max_rows` to `None` on millions of rows), which can crash their browser.
- **Emphasize presentation layout**: Show how formatting numbers (like currency symbols and decimal places) improves the readability of reports and presentations.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `pd.set_option()`, `pd.get_option()`, and `pd.reset_option()` to configure display settings.
- Adjust `display.max_rows` and `display.max_columns` to control output dimensions.
- Use `pd.option_context()` to apply display settings temporarily within a code block.
- Format float values globally using `display.float_format` to standardize decimals and suppress scientific notation.
- Display options only change how data is rendered; they do not modify the underlying values.
