# Resampling, Rolling, and Window Operations

## Lesson Overview

- This chapter explores frequency conversions and window operations on time series data in Pandas. We cover resampling frequencies, rolling window aggregates, expanding calculations, and exponentially weighted moving averages (EWMA).
- Telemetry logs, stock trades, and sales reports are recorded at varying levels of detail. To extract trends or smooth out noise, you must adjust the frequency (e.g. summarizing daily transactions into monthly totals) or compute moving averages (e.g. a 7-day rolling average of active users).
- We will cover `df.resample()` for changing data frequencies, `df.rolling()` for moving window calculations, `df.expanding()` for cumulative running totals, and `df.ewm()` for exponential weighting.
- Mastering these window operations is a key requirement for financial analysis and IoT telemetry analytics.

## Learning Objectives

- Change time-series frequencies using `.resample()` to perform downsampling (aggregation) and upsampling (interpolation).
- Compute moving statistics (mean, sum, std) over fixed time intervals using `.rolling()`.
- Set the `min_periods` parameter to prevent unnecessary `NaN` propagation at the beginning of rolling windows.
- Track cumulative run metrics across entire historical timelines using `.expanding()`.
- Apply exponential smoothing to time series data using `.ewm()` to give greater weight to recent observations.

---

## 1. Resampling Time Series with `.resample()`

The `.resample()` method changes the frequency of a DatetimeIndexed DataFrame. It behaves similarly to `.groupby()`, requiring an aggregation or interpolation function to combine values.

### Downsampling (High to Low Frequency)
Downsampling reduces the frequency of the dataset (e.g. converting hourly readings to daily averages), aggregating multiple rows into one.

```python
import pandas as pd
import numpy as np

# Generate daily sales logs for two weeks
dates = pd.date_range(start="2026-06-01", periods=14, freq="D")
df_daily = pd.DataFrame({"Sales": [100, 150, 120, 200, 180, 250, 300, 110, 130, 140, 220, 210, 260, 310]}, index=dates)

# Resample from Daily to Weekly, calculating total sales per week
weekly_sales = df_daily.resample("W").sum()

print("--- Weekly Consolidated Sales ---")
print(weekly_sales)
```

### Output

```text
--- Weekly Consolidated Sales ---
            Sales
2026-06-07   1120
2026-06-14   1380
```
*Note: The resulting index labels correspond to the end of each weekly period.*

---

### Upsampling (Low to High Frequency)
Upsampling increases the frequency of the dataset (e.g. converting monthly sales to daily estimates), requiring interpolation to fill in the new rows.

```python
# Monthly stock values
df_monthly = pd.DataFrame({"Valuation": [1000, 1200]}, index=pd.to_datetime(["2026-06-01", "2026-07-01"]))

# Upsample to daily frequency using forward fill
df_daily_upsampled = df_monthly.resample("D").ffill()

print("--- Upsampled Daily Valuation (Sample) ---")
print(df_daily_upsampled.head(5))
```

### Output

```text
--- Upsampled Daily Valuation (Sample) ---
            Valuation
2026-06-01       1000
2026-06-02       1000
2026-06-03       1000
2026-06-04       1000
2026-06-05       1000
```

---

## 2. Rolling Window Operations with `.rolling()`

`.rolling()` computes statistics over a moving window of a specified size.

```python
# Calculate a 3-day rolling average of sales
df_daily["Rolling_Mean_3D"] = df_daily["Sales"].rolling(window=3).mean()

# Calculate a 3-day rolling average, requiring at least 1 valid observation
df_daily["Rolling_Mean_Filled"] = df_daily["Sales"].rolling(window=3, min_periods=1).mean()

print("--- Rolling Average Comparisons ---")
print(df_daily.head(6))
```

### Output

```text
--- Rolling Average Comparisons ---
            Sales  Rolling_Mean_3D  Rolling_Mean_Filled
2026-06-01    100              NaN           100.000000
2026-06-02    150              NaN           125.000000
2026-06-03    120       123.333333           123.333333
2026-06-04    200       156.666667           156.666667
2026-06-05    180       166.666667           166.666667
2026-06-06    250       210.000000           210.000000
```
*Note: Without `min_periods=1`, the first 2 rows return `NaN` because the rolling window requires 3 observations to compute the mean.*

---

## 3. Expanding Window Operations with `.expanding()`

`.expanding()` calculates cumulative running statistics (e.g. running totals or running maximums) over all preceding historical data. Unlike rolling windows, the window size grows with each row.

```python
# Calculate the cumulative running sum of sales
df_daily["Cumulative_Sales"] = df_daily["Sales"].expanding().sum()

# Calculate the cumulative running maximum of sales
df_daily["Cumulative_Max"] = df_daily["Sales"].expanding().max()

print("--- Expanding Window Operations ---")
print(df_daily[["Sales", "Cumulative_Sales", "Cumulative_Max"]].head(6))
```

### Output

```text
--- Expanding Window Operations ---
            Sales  Cumulative_Sales  Cumulative_Max
2026-06-01    100             100.0           100.0
2026-06-02    150             250.0           150.0
2026-06-03    120             370.0           150.0
2026-06-04    200             570.0           200.0
2026-06-05    180             750.0           200.0
2026-06-06    250            1000.0           250.0
```

---

## 4. Exponentially Weighted Moving Average (EWMA) with `.ewm()`

Exponential smoothing assigns exponentially decreasing weights to older observations. This allows the moving average to respond more quickly to recent changes in the data compared to a simple rolling average.

```python
# Calculate exponentially weighted moving average with a span of 3 days
df_daily["EWMA_3D"] = df_daily["Sales"].ewm(span=3, adjust=False).mean()

print("--- Simple Rolling vs EWMA ---")
print(df_daily[["Sales", "Rolling_Mean_3D", "EWMA_3D"]].head(6))
```

### Output

```text
--- Simple Rolling vs EWMA ---
            Sales  Rolling_Mean_3D     EWMA_3D
2026-06-01    100              NaN  100.000000
2026-06-02    150              NaN  125.000000
2026-06-03    120       123.333333  122.500000
2026-06-04    200       156.666667  161.250000
2026-06-05    180       166.666667  170.625000
2026-06-06    250       210.000000  210.312500
```

---

## Common Mistakes Students Make

- **Using `.resample()` without an aggregation method**: Running `df.resample('D')` returns a `DatetimeIndexResampler` object. You must call an aggregation function (like `.mean()`, `.sum()`) or interpolation method (like `.ffill()`) to return a DataFrame.
- **Forgetting `min_periods` in rolling windows**: When calculating rolling statistics over large windows (e.g. a 30-day moving average), the first 29 rows will return `NaN`. If you want to compute statistics for these initial rows using the available data, set `min_periods=1`.
- **Misunderstanding the default label behavior in downsampling**: When downsampling to a frequency like monthly (`'M'`), Pandas sets the index label to the end of the month. Use `'MS'` to set the label to the start of the month.
- **Applying rolling windows to unsorted time series**: Running `.rolling()` on an unsorted DatetimeIndex produces incorrect moving averages because the window operates on adjacent rows in the DataFrame, not chronological dates. Always sort the index first.

---

## Best Practices

- Always sort the DatetimeIndex before executing resampling or rolling window operations.
- Use `min_periods=1` in rolling operations to prevent missing values at the beginning of the series.
- Use `.resample()` to standardize time series frequencies before merging datasets from different sources.
- Prefer `.ewm()` over simple rolling averages when analyzing rapidly changing signals (like sensor feeds or trading volumes).

---

## Worked Real-World Examples

### Worked Example 1: Web Traffic Smoothing

```python
import pandas as pd

# Daily website hits
traffic = pd.DataFrame({
    "Hits": [1200, 1500, 1100, 2500, 2800, 3100, 3500, 1400, 1600, 1300, 2200, 2400, 2900, 3200]
}, index=pd.date_range("2026-06-01", periods=14, freq="D"))

# 1. Calculate a 7-day rolling average to smooth out weekend drops
traffic["Rolling_7D"] = traffic["Hits"].rolling(window=7, min_periods=1).mean()

# 2. Resample to weekly totals
weekly_traffic = traffic["Hits"].resample("W").sum()

print("--- Daily and Smoothed Web Traffic (Sample) ---")
print(traffic.head(8))
print("\n--- Weekly Web Traffic ---")
print(weekly_traffic)
```

### Output

```text
--- Daily and Smoothed Web Traffic (Sample) ---
            Hits   Rolling_7D
2026-06-01  1200  1200.000000
2026-06-02  1500  1350.000000
2026-06-03  1100  1266.666667
2026-06-04  2500  1575.000000
2026-06-05  2800  1820.000000
2026-06-06  3100  2033.333333
2026-06-07  3500  2242.857143
2026-06-08  1400  2271.428571

--- Weekly Web Traffic ---
2026-06-07    15700
2026-06-14    14900
Freq: W-SUN, Name: Hits, dtype: int64
```

---

### Worked Example 2: Financial Portfolio Cumulative Peak Tracker

```python
import pandas as pd

# Monthly portfolio valuation
portfolio = pd.DataFrame({
    "Value": [10000, 11500, 10800, 12500, 12100, 13000]
}, index=pd.date_range("2026-01-01", periods=6, freq="MS"))

# 1. Track cumulative peak value
portfolio["Peak_Value"] = portfolio["Value"].expanding().max()

# 2. Calculate Drawdown percentage from peak
portfolio["Drawdown_Pct"] = ((portfolio["Value"] - portfolio["Peak_Value"]) / portfolio["Peak_Value"]) * 100

print("--- Portfolio Peak and Drawdowns ---")
print(portfolio)
```

### Output

```text
--- Portfolio Peak and Drawdowns ---
            Value  Peak_Value  Drawdown_Pct
2026-01-01  10000       10000      0.000000
2026-02-01  11500       11500      0.000000
2026-03-01  10800       11500     -6.086957
2026-04-01  12500       12500      0.000000
2026-05-01  12100       12500     -3.200000
2026-06-01  13000       13000      0.000000
```

---

### Worked Example 3: Upsampling Temperature Telemetry

```python
import pandas as pd

# Irregular 2-hour sensor logs
temp_logs = pd.DataFrame({
    "Temp": [22.5, 23.8, 24.2]
}, index=pd.to_datetime(["2026-06-01 08:00", "2026-06-01 10:00", "2026-06-01 12:00"]))

# Upsample to hourly frequency and interpolate values linearly
hourly_temp = temp_logs.resample("H").interpolate(method="linear")

print("--- Interpolated Hourly Temperature ---")
print(hourly_temp)
```

### Output

```text
--- Interpolated Hourly Temperature ---
                      Temp
2026-06-01 08:00:00  22.50
2026-06-01 09:00:00  23.15
2026-06-01 10:00:00  23.80
2026-06-01 11:00:00  24.00
2026-06-01 12:00:00  24.20
```

---

## Practice Questions

1. Explain the differences between downsampling and upsampling inside the `.resample()` method.
2. Write a command to downsample daily data to monthly averages.
3. Compare the behavior of `.rolling()` and `.expanding()` moving windows.
4. Write a command to calculate a 7-day rolling standard deviation of a Series `s`, requiring at least 3 valid observations.
5. Explain the purpose and mathematical advantages of the Exponentially Weighted Moving Average (`.ewm()`).
6. Write a command to upsample weekly stock prices to daily frequency, using backward filling to fill missing values.
7. How does the `min_periods` parameter inside `.rolling()` prevent unnecessary `NaN` propagation?
8. Write a script to calculate the cumulative running maximum of a Series.
9. Explain the default label indexing behavior of `.resample()` for downsampling.
10. Describe how to perform linear interpolation during upsampling in Pandas.

---

## Mini Assignments

### Assignment 1: Stock Price Moving Average Crossover
- Create a stock price DataFrame with daily closing prices for 30 trading days.
- Calculate the 5-day and 20-day simple rolling averages.
- Add a column indicating whether the 5-day average is above the 20-day average.

### Assignment 2: Server Temperature Telemetry Resampling
- Create a sensor log DataFrame tracking temperature readings at 10-minute intervals.
- Resample the data to hourly averages and hourly maximums.
- Identify hours where the maximum temperature exceeded a specific threshold.

### Assignment 3: Expanding Sales Budget Tracker
- Create a Series tracking monthly sales revenue for a year.
- Calculate the cumulative running sum of sales.
- Calculate the percentage contribution of each month's sales to the cumulative total at that point in time.

---

## Interview-Oriented Questions

- **What is the difference between `.rolling()` and `.expanding()` in Pandas?**
  - *Answer*: `.rolling()` uses a fixed window size that slides along the index, calculating statistics only on the elements inside the window. `.expanding()` uses a growing window that starts at the first element and expands to include all cumulative historical data up to the current row.
- **Explain the role of the `min_periods` parameter in rolling window calculations.**
  - *Answer*: `min_periods` defines the minimum number of valid (non-null) observations required inside the window to return a calculated statistic. By default, it equals the window size, meaning the initial rows of the series will return `NaN`. Setting `min_periods=1` allows statistics to be calculated for these initial rows using whatever data is available in the window.
- **How does upsampling handle missing values by default, and how can we interpolate them?**
  - *Answer*: Upsampling inserts new rows for the higher frequency and fills the cells with `NaN` by default. To fill these missing values, chain an interpolation method (like `.ffill()`, `.bfill()`, `.interpolate(method='linear')`, or `.asfreq()`) to the resampler object.
- **Explain how Exponentially Weighted Moving Average (`.ewm()`) reduces lag in trend detection.**
  - *Answer*: Unlike simple moving averages that assign equal weight to all points in the window, EWMA assigns exponentially decreasing weights to older observations. This gives recent data points greater influence, allowing the average to respond more quickly to trend changes and reducing lag.
- **Why must a time series DataFrame be sorted before running window calculations?**
  - *Answer*: Window calculations operate on adjacent rows in the DataFrame. If the DatetimeIndex is unsorted, the window will combine non-chronological records, producing incorrect moving averages or trend summaries.

---

## Teaching Notes for This Chapter

- **Visualize Window Calculations**: Draw a time series line on the board and illustrate how a rolling window slides along the curve, while an expanding window accumulates data.
- **Demonstrate Interpolation Options**: Compare forward fill, backward fill, and linear interpolation on a whiteboard to help students understand upsampling options.
- **Emphasize the Resampler object**: Remind students that `.resample()` behaves like `.groupby()` and requires a terminal aggregation method to return data.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `.resample()` to change time-series frequencies.
- Downsampling aggregates high-frequency data to a lower frequency; upsampling increases frequency and requires interpolation (like `.ffill()` or `.interpolate()`).
- `.rolling(window)` computes statistics over a moving window of a fixed size.
- Set `min_periods=1` in rolling operations to prevent missing values at the start of the series.
- `.expanding()` computes running statistics over all preceding historical data.
- `.ewm(span)` calculates exponentially weighted statistics, giving more weight to recent observations and reducing lag.
- Always sort the DatetimeIndex before running resampling or window operations.
