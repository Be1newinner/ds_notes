# Integrating Pandas with Visualization Libraries (Matplotlib, Seaborn)

## Lesson Overview

- This chapter covers data visualization in Pandas. We explore the built-in `.plot()` wrapper, customizing charts using Matplotlib, and generating statistical plots (like boxplots, pairplots, and correlation heatmaps) using Seaborn.
- Raw tables of numbers are hard to interpret. Visualizing distributions, trends, and relationships using charts is essential for communicating insights and verifying patterns.
- We will cover built-in plotting wrappers, customize axes and layouts using Matplotlib, and use Seaborn to generate distribution plots and heatmaps.
- Mastering these plotting tools allows you to create publication-grade charts directly from your DataFrames.

## Learning Objectives

- Generate quick exploratory charts directly from DataFrames using the built-in `.plot()` wrapper.
- Customize labels, legends, and gridlines using the core Matplotlib API.
- Create statistical distribution plots (boxplots, histograms, KDEs) using Seaborn.
- Visualize relationships and correlations using Seaborn pairplots and heatmaps.
- Export high-resolution charts to files using `plt.savefig()`.

---

## 1. Built-in Plotting with `df.plot()`

Pandas has a built-in `.plot()` method that wraps Matplotlib, allowing you to generate charts directly from Series and DataFrames.

### Key Parameters
- **`kind`**: `'line'` (default), `'bar'`, `'barh'` (horizontal bar), `'hist'`, `'box'`, `'scatter'`, `'kde'`.
- **`x` / `y`**: Column names to plot along the axes.
- **`title`**: Chart title.
- **`figsize`**: Tuple defining chart dimensions (width, height in inches).

### Setup for Demonstration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame representing company performance
df_perf = pd.DataFrame({
    "Year": [2022, 2023, 2024, 2025, 2026],
    "Revenue": [120, 150, 180, 240, 310],
    "Profit": [15, 22, 28, 42, 58],
    "Marketing_Spend": [10, 12, 18, 25, 30]
})

print("--- Master Performance Table ---")
print(df_perf)
```

### Output

```text
--- Master Performance Table ---
   Year  Revenue  Profit  Marketing_Spend
0  2022      120      15               10
1  2023      150      22               12
2  2024      180      28               18
3  2025      240      42               25
4  2026      310      58               30
```

---

### Generating a Line Plot

```python
# Generate a line plot comparing Revenue and Profit over the years
df_perf.plot(
    x="Year",
    y=["Revenue", "Profit"],
    kind="line",
    title="Corporate Performance (2022-2026)",
    figsize=(8, 4),
    marker="o"
)

# Customize using Matplotlib APIs
plt.xlabel("Year")
plt.ylabel("Amount (Millions USD)")
plt.grid(True, linestyle="--", alpha=0.5)

# Save the plot
plt.savefig("corporate_performance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Line plot generated and saved as 'corporate_performance.png'.")
```

### Output

```text
Line plot generated and saved as 'corporate_performance.png'.
```

---

## 2. Statistical Visualizations with Seaborn

Seaborn is a high-level visualization library built on top of Matplotlib. It integrates with Pandas, automatically parsing DataFrame headers to define chart labels, legends, and groupings.

### Setup for Demonstration

```python
# User session log
df_sessions = pd.DataFrame({
    "Device": ["Mobile", "Desktop", "Mobile", "Mobile", "Desktop", "Mobile", "Desktop", "Desktop"],
    "Age": [22, 35, 19, 28, 45, 24, 38, 50],
    "Purchase_Amt": [120, 350, 80, 150, 420, 95, 280, 500]
})
```

---

### A. Distribution Plots: Boxplots and Histograms

Boxplots are ideal for visualizing the distribution of a numeric column across categories, highlighting medians and outliers.

```python
# Generate a boxplot comparing Purchase Amount by Device type
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_sessions, x="Device", y="Purchase_Amt", palette="Set2")

plt.title("Purchase Amount Distribution by Device")
plt.xlabel("Device Category")
plt.ylabel("Purchase Amount ($)")

plt.savefig("purchase_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Boxplot generated and saved as 'purchase_distribution.png'.")
```

### Output

```text
Boxplot generated and saved as 'purchase_distribution.png'.
```

---

### B. Correlation Heatmaps

A correlation heatmap visualizes relationships between multiple numeric variables in a grid.

```python
# Calculate the correlation matrix
correlation_matrix = df_perf.corr()

# Generate the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title("Performance Correlation Matrix")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Correlation heatmap generated and saved as 'correlation_heatmap.png'.")
```

### Output

```text
Correlation heatmap generated and saved as 'correlation_heatmap.png'.
```

---

### C. Pairwise Relationships: Pairplots

Seaborn's `pairplot()` generates a grid of pairwise scatter plots and histograms across all numeric columns in a DataFrame, showing relationships between variables.

```python
# Generate a pairplot grouped by Device category
sns_pair = sns.pairplot(df_sessions, hue="Device", palette="husl")
sns_pair.savefig("sessions_pairplot.png", dpi=300)
plt.close()
print("Pairplot generated and saved as 'sessions_pairplot.png'.")
```

### Output

```text
Pairplot generated and saved as 'sessions_pairplot.png'.
```

---

## Common Mistakes Students Make

- **Forgetting `plt.close()` in scripts**: When generating multiple plots in a script or loop, forgetting to call `plt.close()` can cause memory leaks because Matplotlib keeps all figure objects in memory.
- **Passing lists instead of DataFrame columns**: Writing `plt.scatter(df, x, y)` instead of `df.plot(kind='scatter', x='ColA', y='ColB')` is a common syntax error. Matplotlib expects Series, whereas Seaborn and Pandas plot wrappers accept DataFrame column names directly.
- **Ignoring layout clipping**: Text labels and titles can be cut off when saving figures. Always use `bbox_inches='tight'` inside `plt.savefig()` to ensure all elements are saved correctly.
- **Attempting to plot non-numeric columns**: Trying to generate line plots or histograms on columns containing text strings raises a `TypeError`. Filter out text columns before plotting.

---

## Best Practices

- Always use `bbox_inches='tight'` in `plt.savefig()` to prevent chart elements from being clipped in saved files.
- Use `sns.set_theme()` at the beginning of visualization scripts to apply standard, clean plotting styles.
- Specify the `hue` parameter in Seaborn plots to group data by categorical columns (like regions or groups) automatically.
- Save figures using vector formats (like PDF or SVG) or high-resolution PNGs (`dpi=300`) for presentations and reports.

---

## Worked Real-World Examples

### Worked Example 1: Monthly Customer Churn Risk Plotting

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Client logs
clients = pd.DataFrame({
    "Tier": ["Gold", "Silver", "Bronze", "Silver", "Gold", "Bronze", "Silver", "Bronze"],
    "Support_Tickets": [1, 4, 8, 3, 2, 9, 5, 7],
    "Payment_Delay_Days": [2, 12, 15, 8, 1, 20, 10, 18]
})

# Generate a scatter plot comparing Support Tickets and Payment Delay Days grouped by Tier
plt.figure(figsize=(7, 4))
sns.scatterplot(
    data=clients,
    x="Support_Tickets",
    y="Payment_Delay_Days",
    hue="Tier",
    style="Tier",
    s=100
)

plt.title("Customer Churn Risk Analysis")
plt.xlabel("Support Tickets Raised")
plt.ylabel("Payment Delay (Days)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("churn_risk_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("Scatter plot generated and saved as 'churn_risk_analysis.png'.")
```

### Output

```text
Scatter plot generated and saved as 'churn_risk_analysis.png'.
```

---

### Worked Example 2: Financial Sales Contribution Bar Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

# Regional sales total
sales_data = pd.DataFrame({
    "Revenue": [120000, 85000, 95000]
}, index=["North_Region", "South_Region", "West_Region"])

# Generate a horizontal bar chart
sales_data.plot(
    kind="barh",
    color="skyblue",
    legend=False,
    figsize=(7, 3)
)

plt.title("Regional Sales Contribution")
plt.xlabel("Revenue (USD)")
plt.ylabel("Region")
plt.axvline(100000, color="red", linestyle="--", label="Target Threshold")
plt.legend()

plt.savefig("regional_sales.png", dpi=300, bbox_inches="tight")
plt.close()
print("Horizontal bar chart saved as 'regional_sales.png'.")
```

### Output

```text
Horizontal bar chart saved as 'regional_sales.png'.
```

---

## Practice Questions

1. Explain the differences between using the built-in `.plot()` wrapper versus calling Matplotlib APIs directly.
2. Write a command to generate a histogram of column `Sales` with 15 bins.
3. How does the `hue` parameter inside Seaborn methods change the structure of a chart?
4. Write a command to generate a Seaborn correlation heatmap using a custom color palette.
5. Explain the purpose and parameters of the `plt.savefig()` method.
6. Write a command to generate a boxplot comparing `Salary` (y-axis) across `Department` (x-axis) categories.
7. What error occurs when you attempt to plot text columns using `df.plot(kind='line')`?
8. Write a script to display a grid of pairwise plots for a DataFrame using Seaborn.
9. How can you set a title and customize axis labels for a chart generated using `.plot()`?
10. Describe how to plot a cumulative line chart using cumulative sum values.

---

## Mini Assignments

### Assignment 1: Corporate Performance Line Chart
- Create a performance DataFrame tracking `Quarter` (Q1 to Q4), `Sales`, and `Expenses` for a year.
- Generate a line chart comparing sales and expenses, customizing labels, legends, and gridlines.
- Save the chart as a high-resolution PNG.

### Assignment 2: Customer Demographics Boxplot
- Create a customer demographics dataset tracking `Gender`, `Age`, and `Spend_Amount`.
- Generate a Seaborn boxplot comparing the distribution of `Spend_Amount` across `Gender` categories.
- Export the chart to a file.

### Assignment 3: Financial Correlation Heatmap
- Create a DataFrame containing 4 numeric variables (e.g. Stock Price, Interest Rate, Volume, Inflation).
- Generate a Seaborn correlation heatmap using a diverging color palette (like `'coolwarm'`), displaying correlation values inside each cell.

---

## Interview-Oriented Questions

- **What is the difference between Matplotlib and Seaborn visualization libraries?**
  - *Answer*: Matplotlib is a low-level plotting library that provides complete control over every element in a figure, but requires verbose code to generate complex charts. Seaborn is a high-level library built on top of Matplotlib. It integrates with Pandas, automatically parsing DataFrame headers to define styles, labels, and legends with minimal code.
- **Explain the role of the `ax` parameter inside the `.plot()` method.**
  - *Answer*: The `ax` parameter allows you to specify a Matplotlib Axes object on which to render the chart. This is useful for creating subplots, layering multiple charts on the same figure, or integrating Pandas plots into larger custom Matplotlib layouts.
- **How does Seaborn handle missing values (`NaN`) when generating statistical plots?**
  - *Answer*: Seaborn ignores missing values when calculating statistical summaries (like means or distributions) and rendering charts. For example, in scatter plots, points with `NaN` coordinates are omitted.
- **Why is it recommended to specify `bbox_inches='tight'` when saving figures?**
  - *Answer*: By default, `plt.savefig()` uses a fixed bounding box that can clip labels, legends, or titles if they extend past the default figure borders. `bbox_inches='tight'` calculates the bounding box dynamically to ensure all elements are saved correctly.
- **How can we create a multi-panel subplot grid using Matplotlib?**
  - *Answer*: Use the `plt.subplots(nrows, ncols)` function. This returns a Figure object and an array of Axes objects. You can then specify which axis to plot on using the `ax` parameter: `df.plot(ax=axes[0, 1])`.

---

## Teaching Notes for This Chapter

- **Deconstruct Subplots**: Draw a grid of subplots on a whiteboard to illustrate how Axes coordinates work.
- **Showcase Theme Customizations**: Compare the default Matplotlib style with Seaborn themes in class to illustrate how design affects readability.
- **Emphasize vector export formats**: Encourage students to save figures in vector formats (like PDF or SVG) for print publications to preserve quality.

---

## Chapter Wrap-up Concepts Students Must Master

- Use the built-in `.plot()` method to generate quick exploratory charts directly from DataFrames.
- Customize labels, legends, and gridlines using the core Matplotlib API.
- Use Seaborn to generate statistical distribution plots (like boxplots and histograms) and correlation heatmaps.
- Specify the `hue` parameter in Seaborn to group data by categorical columns automatically.
- Save figures using `plt.savefig()` with `bbox_inches='tight'` to prevent element clipping.
- Call `plt.close()` in scripts to release memory when generating multiple plots.
