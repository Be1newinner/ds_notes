# Exporting and Reporting (CSV, Excel, JSON, HTML)

## Lesson Overview

- This chapter covers exporting DataFrames to various file formats and applying style formatting to generate reports in Pandas.
- Getting insights from data is only half the battle; you must share your findings with stakeholders in suitable formats—such as raw CSVs, multi-sheet Excel workbooks, structured JSON payloads, or styled HTML tables.
- We will cover `.to_csv()`, `.to_excel()` with `pd.ExcelWriter`, `.to_json()` orientations, `.to_html()`, and styling tables using the `df.style` API.
- Mastering these export and styling tools allows you to generate executive-ready reports directly from your code.

## Learning Objectives

- Export DataFrames to CSV format, managing index preservation and UTF-8 encodings.
- Write multi-sheet Excel workbooks using `pd.ExcelWriter`.
- Restructure DataFrames into JSON outputs using standard orientations (records, split, index).
- Generate HTML tables from DataFrames for browser displays.
- Highlight values and format data elements using the `df.style` API.

---

## 1. Exporting to CSV and Text Formats with `.to_csv()`

`.to_csv()` exports DataFrames to comma-separated text files.

### Key Parameters
- **`index`**: `False` to omit row index numbers in the output file.
- **`sep`**: Character delimiter (defaults to `','`).
- **`encoding`**: Encoding scheme (defaults to `'utf-8'`).
- **`na_rep`**: String representation for missing values.

### Setup for Demonstration

```python
import pandas as pd

# Corporate performance report
df_report = pd.DataFrame({
    "Region": ["East", "West", "North"],
    "Revenue": [120000.50, 95000.00, 110000.00],
    "Margin": [0.18, 0.12, 0.15]
})

# Export to CSV omitting the row index
df_report.to_csv("regional_report.csv", index=False, na_rep="NaN")
print("Report exported to 'regional_report.csv'.")
```

### Output

```text
Report exported to 'regional_report.csv'.
```

---

## 2. Multi-Sheet Excel Exports with `pd.ExcelWriter`

To export multiple DataFrames to separate sheets within the same Excel file, use the **`pd.ExcelWriter`** context manager. This requires libraries like `openpyxl` or `xlsxwriter` to be installed.

```python
# Regional segments
df_east = pd.DataFrame({"Item": ["Laptop"], "Qty": [15]})
df_west = pd.DataFrame({"Item": ["Mouse"], "Qty": [42]})

# Write both segments to the same Excel file
with pd.ExcelWriter("corporate_segments.xlsx", engine="openpyxl") as writer:
    df_east.to_excel(writer, sheet_name="East_Region", index=False)
    df_west.to_excel(writer, sheet_name="West_Region", index=False)

print("Excel file 'corporate_segments.xlsx' generated successfully.")
```

### Output

```text
Excel file 'corporate_segments.xlsx' generated successfully.
```

---

## 3. Formatting JSON Outputs with `.to_json()`

JSON is a standard format for web integration. The **`orient`** parameter determines the layout of the output string:
- **`records`**: List of row objects (preferred for web APIs).
- **`split`**: Separate dictionary lists for index, columns, and data.
- **`index`**: Nested dictionary keyed by row index.
- **`columns`**: Nested dictionary keyed by column name.

```python
# Convert to JSON using different orientations
json_records = df_report.to_json(orient="records", indent=2)
json_split = df_report.to_json(orient="split", indent=2)

print("--- JSON Records Orientation ---")
print(json_records)
```

### Output

```text
--- JSON Records Orientation ---
[
  {
    "Region":"East",
    "Revenue":120000.5,
    "Margin":0.18
  },
  {
    "Region":"West",
    "Revenue":95000.0,
    "Margin":0.12
  },
  {
    "Region":"North",
    "Revenue":110000.0,
    "Margin":0.15
  }
]
```

---

## 4. Generating Browser-Ready HTML Tables with `.to_html()`

Use `.to_html()` to convert a DataFrame to an HTML `<table>` string for web dashboards or email reports.

```python
# Export to HTML string
html_table = df_report.to_html(classes="table table-striped", index=False)
print("--- HTML Table Class String (Sample) ---")
print(html_table[:150] + "\n...")
```

### Output

```text
--- HTML Table Class String (Sample) ---
<table border="1" class="dataframe table table-striped">
  <thead>
    <tr style="text-align: right;">
      <th>Region</th>
      <th>Revenue
...
```

---

## 5. Styling Outputs using the `df.style` API

The `df.style` property returns a Styler object that allows you to apply conditional formatting and styling to DataFrame displays.

```python
# Format revenue as currency and margin as percentage
# Highlight the maximum values in each column in light green
styled_report = (
    df_report.style
    .format({"Revenue": "${:,.2f}", "Margin": "{:.1%}"})
    .highlight_max(color="lightgreen", axis=0)
)

print("--- Styled Report object representation ---")
print(type(styled_report))
```

### Output

```text
--- Styled Report object representation ---
<class 'pandas.io.formats.style.Styler'>
```
*Note: In Jupyter Notebooks, the Styler object renders directly as a styled HTML table.*

---

## Common Mistakes Students Make

- **Unintentionally saving row indexes in CSV files**: Forgetting to set `index=False` inside `.to_csv()` saves the row numbers as an unnamed first column in the CSV file, which can cause import issues later.
- **Accidentally overwriting Excel sheets**: Calling `.to_excel()` on the same file path multiple times without using `pd.ExcelWriter(..., mode='a')` overwrites the file, deleting previously saved sheets.
- **Using Styler objects for calculations**: The `df.style` API returns a Styler object, not a DataFrame. Attempting to perform calculations on a Styler object (e.g. `styled_df.mean()`) raises an `AttributeError`. Apply styles only at the very end of your script.
- **Ignoring encoding issues for special characters**: Saving datasets containing special characters (like currency symbols or non-English letters) using default system encodings can cause corruption. Always specify `encoding='utf-8'` to preserve characters.

---

## Best Practices

- Always set `index=False` in `.to_csv()` and `.to_excel()` unless the row index contains descriptive labels.
- Use the `pd.ExcelWriter` context manager when writing multiple sheets to the same Excel file.
- Use `orient='records'` in `.to_json()` when exporting data for web APIs or databases.
- Apply formatting styles only at the end of your analytical script, as styling methods return a Styler object rather than a DataFrame.

---

## Worked Real-World Examples

### Worked Example 1: Executive Multi-Sheet Sales Report

```python
import pandas as pd

# Region records
df_sales = pd.DataFrame({"Region": ["East", "West"], "Revenue": [120000, 95000]})
df_expenses = pd.DataFrame({"Region": ["East", "West"], "Marketing": [15000, 12000], "Rent": [5000, 5000]})

# Write data to separate sheets in the corporate workbook
with pd.ExcelWriter("corporate_finance.xlsx", engine="openpyxl") as writer:
    df_sales.to_excel(writer, sheet_name="Sales_Report", index=False)
    df_expenses.to_excel(writer, sheet_name="Expense_Report", index=False)

print("Corporate financial workbook generated successfully.")
```

### Output

```text
Corporate financial workbook generated successfully.
```

---

### Worked Example 2: Interactive Styled Web Report

```python
import pandas as pd

# Regional performance records
performance = pd.DataFrame({
    "Branch": ["Delhi", "Mumbai", "Kolkata"],
    "Target_Met": [True, False, True],
    "Revenue_USD": [150000, 85000, 110000]
})

# Format columns and color code the target status
styled_report = (
    performance.style
    .format({"Revenue_USD": "${:,.2f}"})
    .highlight_max(subset=["Revenue_USD"], color="lightgreen")
)

# Export the styled table directly to HTML
styled_report.to_html("styled_report.html")
print("Styled HTML report exported to 'styled_report.html'.")
```

### Output

```text
Styled HTML report exported to 'styled_report.html'.
```

---

## Practice Questions

1. Explain the default behavior of the `index` parameter inside `.to_csv()` and why we set it to `False`.
2. Write a command to export a DataFrame `df` to a tab-separated text file named `data.txt`.
3. How does `pd.ExcelWriter` allow writing to multiple sheets in a single file?
4. Write a command to convert a DataFrame `df` into a list of row dictionaries (JSON format).
5. Explain the differences between the `records`, `split`, and `columns` JSON orientations.
6. Write a command to export a DataFrame directly to a styled HTML table with a custom class attribute.
7. What object type is returned when applying cell styling (e.g. `.highlight_max()`) to a DataFrame?
8. Write a script that saves a DataFrame to an Excel sheet, starting the data layout at row 2 and column B.
9. How can you set a custom string (e.g. `'MISSING'`) to represent null values in a CSV export?
10. Describe how to apply a background color gradient to columns in a DataFrame using the Styler API.

---

## Mini Assignments

### Assignment 1: Corporate Segment Excel Workbook
- Create three segment DataFrames: `df_tech`, `df_hr`, and `df_finance` (3 rows each).
- Write these DataFrames to separate sheets (`"Tech_Staff"`, `"HR_Staff"`, `"Finance_Staff"`) within an Excel workbook named `personnel.xlsx`.

### Assignment 2: REST API JSON Payload Export
- Create a client DataFrame containing `ClientID`, `Email`, and `Registration_Date`.
- Convert the DataFrame into a JSON string using `orient='records'` with an indentation level of 4.
- Save the string to a text file named `payload.json`.

### Assignment 3: Executive Dashboard styled HTML
- Create a regional financial report tracking `Region`, `Q1_Revenue`, and `Q2_Revenue`.
- Format the revenues as currency.
- Highlight the minimum revenue in red and the maximum revenue in green.
- Export the styled report to an HTML file named `dashboard.html`.

---

## Interview-Oriented Questions

- **Why is setting `index=False` standard practice when exporting DataFrames to CSV?**
  - *Answer*: By default, `.to_csv()` saves the DataFrame row numbers as an unnamed first column in the output file. When importing this file later, Pandas loads these row numbers as a regular column, resulting in duplicate or unnecessary columns. Setting `index=False` prevents this.
- **Explain the difference between `orient='records'` and `orient='split'` in `.to_json()`.**
  - *Answer*: `orient='records'` outputs a list of dictionaries where each dictionary represents a row (keyed by column name). `orient='split'` outputs a single dictionary containing three lists: `columns`, `index`, and `data`. This separates the metadata from the cell values, reducing file size.
- **How can we append a new sheet to an existing Excel file without overwriting the previous sheets?**
  - *Answer*: Use `pd.ExcelWriter` with `mode='a'` (append). This requires the `'openpyxl'` engine: `with pd.ExcelWriter('file.xlsx', mode='a', engine='openpyxl') as writer: df.to_excel(writer, sheet_name='New_Sheet')`.
- **What is the `Styler` object in Pandas, and can we perform statistical calculations on it?**
  - *Answer*: The `Styler` object (accessed via `df.style`) is a wrapper that formats DataFrames for display. It cannot be used for calculations; attempting to call DataFrame methods on it raises an `AttributeError`. Apply styling only at the very end of your script.
- **How does the `encoding` parameter in `.to_csv()` prevent data corruption?**
  - *Answer*: The `encoding` parameter determines the character encoding of the output file. Specifying `encoding='utf-8'` ensures that special characters (like currency symbols or non-English letters) are saved correctly, preventing corruption when the file is loaded on other systems.

---

## Teaching Notes for This Chapter

- **Deconstruct JSON Formats**: Compare `records` and `split` JSON layouts on a whiteboard to illustrate structural differences.
- **Demonstrate ExcelWriter**: Walk students through the `ExcelWriter` context manager syntax to prevent overwriting errors in class.
- **Highlight Styler limits**: Remind students that styling methods return a Styler object rather than a DataFrame, helping them avoid calculation errors.

---

## Chapter Wrap-up Concepts Students Must Master

- Set `index=False` in `.to_csv()` and `.to_excel()` to omit row numbers from exports.
- Use `pd.ExcelWriter` to write multiple sheets to the same Excel file.
- Use `to_json(orient='records')` when exporting data for web APIs.
- Use `to_html()` to convert DataFrames to HTML tables for browser displays.
- Apply conditional formatting and styles using the Styler API (`df.style`) at the end of scripts.
- Specify `encoding='utf-8'` to preserve special characters during exports.
