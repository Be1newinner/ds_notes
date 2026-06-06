# Time Series and DateTime Handling

## Lesson Overview

- This chapter covers time series and datetime handling in Pandas.
- Time series analysis is a core application of Pandas in finance, IoT telemetry, log parsing, and business operations. Analyzing trends, seasonal cycles, or lags requires dates to be represented as datetime objects rather than raw strings.
- We will cover creating DatetimeIndexes using `pd.date_range()`, slicing timeline records using partial string indexing, shifting data with `.shift()`, converting timezones, and using the `.dt` accessor.
- Mastering these time series features enables you to perform date calculations and analyze trends.

## Learning Objectives

- Construct chronological timelines using `pd.date_range()` with custom frequencies (daily, business days, hourly).
- Filter and slice time series data using partial string indexing (by year, month, or range).
- Lag and lead dataset elements using `.shift()` to calculate period-over-period changes.
- Handle timezone localizations and conversions using `.tz_localize()` and `.tz_convert()`.
- Extract specific time components (e.g. year, day of week, hour) using the `.dt` accessor.

---

## Representing Time in Pandas

Pandas represents time using two primary classes:
1. **`Timestamp`**: Represents a single point in time (similar to Python's datetime).
2. **`DatetimeIndex`**: A Series or Index of Timestamps, enabling date-based operations.

### Setup for Demonstration

```python
import pandas as pd

# Create datetime series from raw strings
date_series = pd.Series(["2026-06-01 09:00:00", "2026-06-02 10:30:00", "2026-06-03 14:15:00"])

# Convert strings to datetime objects
date_parsed = pd.to_datetime(date_series)

print("--- Parsed Datetime Series ---")
print(date_parsed)
print("Type:", date_parsed.dtype)
```

### Output

```text
--- Parsed Datetime Series ---
0   2026-06-01 09:00:00
1   2026-06-02 10:30:00
2   2026-06-03 14:15:00
dtype: datetime64[ns]
Type: datetime64[ns]
```

---

## 1. Creating Chronological Timelines with `pd.date_range()`

`pd.date_range()` generates a sequence of dates at a specified frequency:
- `D`: Daily (default)
- `B`: Business day (excludes weekends)
- `H`: Hourly
- `M`: Month-end
- `MS`: Month-start
- `T` or `min`: Minutely

```python
# Generate business days in June 2026
business_days = pd.date_range(start="2026-06-01", end="2026-06-05", freq="B")
print("--- Business Days ---")
print(business_days)

# Generate hourly timestamps
hourly_range = pd.date_range(start="2026-06-01 08:00", periods=4, freq="H")
print("\n--- Hourly Intervals ---")
print(hourly_range)
```

### Output

```text
--- Business Days ---
DatetimeIndex(['2026-06-01', '2026-06-02', '2026-06-03', '2026-06-04',
               '2026-06-05'],
              dtype='datetime64[ns]', freq='B')

--- Hourly Intervals ---
DatetimeIndex(['2026-06-01 08:00:00', '2026-06-01 09:00:00',
               '2026-06-01 10:00:00', '2026-06-01 11:00:00'],
              dtype='datetime64[ns]', freq='H')
```

---

## 2. Slicing Time Series via Partial String Indexing

When a DatetimeIndex is set as the DataFrame index, you can slice and filter records using partial date strings.

```python
# Create daily price logs
dates = pd.date_range(start="2026-05-28", periods=8, freq="D")
df_ts = pd.DataFrame({"Price": [100, 101, 102, 103, 104, 105, 106, 107]}, index=dates)

print("--- Master Time Series Table ---")
print(df_ts)

# Filter by a specific month (June 2026)
june_data = df_ts.loc["2026-06"]
print("\n--- June 2026 Data ---")
print(june_data)

# Slice using a date range
range_data = df_ts.loc["2026-05-30":"2026-06-02"]
print("\n--- Sliced Date Range ---")
print(range_data)
```

### Output

```text
--- Master Time Series Table ---
            Price
2026-05-28    100
2026-05-29    101
2026-05-30    102
2026-05-31    103
2026-06-01    104
2026-06-02    105
2026-06-03    106
2026-06-04    107

--- June 2026 Data ---
            Price
2026-06-01    104
2026-06-02    105
2026-06-03    106
2026-06-04    107

--- Sliced Date Range ---
            Price
2026-05-30    102
2026-05-31    103
2026-06-01    104
2026-06-02    105
```

---

## 3. Lagging and Leading Data with `.shift()`

The `.shift()` method moves index values forward or backward by a specified number of periods. This is useful for calculating day-over-day growth rates.

```python
# Add lag column (yesterday's price)
df_ts["Lag_Price"] = df_ts["Price"].shift(1)

# Add lead column (tomorrow's price)
df_ts["Lead_Price"] = df_ts["Price"].shift(-1)

# Calculate Daily growth rate (%)
df_ts["Daily_Return_Pct"] = ((df_ts["Price"] - df_ts["Lag_Price"]) / df_ts["Lag_Price"]) * 100

print("--- Lagged and Calculated Returns ---")
print(df_ts)
```

### Output

```text
--- Lagged and Calculated Returns ---
            Price  Lag_Price  Lead_Price  Daily_Return_Pct
2026-05-28    100        NaN       101.0               NaN
2026-05-29    101      100.0       102.0          1.000000
2026-05-30    102      101.0       103.0          0.990099
2026-05-31    103      102.0       104.0          0.980392
2026-06-01    104      103.0       105.0          0.970874
2026-06-02    105      104.0       106.0          0.961538
2026-06-03    106      105.0       107.0          0.952381
2026-06-04    107      106.0         NaN          0.943396
```

---

## 4. Extracting Datetime Attributes with `.dt`

If a datetime column is stored as a regular column (not set as the index), you can extract time components (like year, day name, or hour) using the **`.dt`** accessor.

```python
# Roster of logs
df_logs = pd.DataFrame({
    "LogID": ["L1", "L2", "L3"],
    "Timestamp": pd.to_datetime(["2026-06-01 08:30:00", "2026-06-01 14:45:00", "2026-06-02 22:15:00"])
})

# Extract datetime properties
df_logs["Hour"] = df_logs["Timestamp"].dt.hour
df_logs["Day_Name"] = df_logs["Timestamp"].dt.day_name()
df_logs["Is_Weekend"] = df_logs["Timestamp"].dt.dayofweek >= 5

print("--- Extracted Datetime Features ---")
print(df_logs)
```

### Output

```text
--- Extracted Datetime Features ---
  LogID           Timestamp  Hour Day_Name  Is_Weekend
0    L1 2026-06-01 08:30:00     8   Monday       False
1    L2 2026-06-01 14:45:00    14   Monday       False
2    L3 2026-06-02 22:15:00    22  Tuesday       False
```

---

## 5. Timezone Localization and Conversion

By default, Pandas datetime objects are timezone-naive. Use `.tz_localize()` to set a timezone, and `.tz_convert()` to convert between timezones.

```python
# Timezone-naive series
naive_time = pd.Series(pd.date_range("2026-06-01 09:00", periods=2, freq="H"))

# 1. Localize to UTC
utc_time = naive_time.dt.tz_localize("UTC")
print("--- Localized UTC ---")
print(utc_time)

# 2. Convert to Indian Standard Time (IST)
ist_time = utc_time.dt.tz_convert("Asia/Kolkata")
print("\n--- Converted to IST ---")
print(ist_time)
```

### Output

```text
--- Localized UTC ---
0   2026-06-01 09:00:00+00:00
1   2026-06-01 10:00:00+00:00
dtype: datetime64[ns, UTC]

--- Converted to IST ---
0   2026-06-01 14:30:00+05:30
1   2026-06-01 15:30:00+05:30
dtype: datetime64[ns, Asia/Kolkata]
```

---

## Common Mistakes Students Make

- **Using `.dt` on the Index**: The `.dt` accessor is only available on Series objects. If the datetime is set as the DataFrame index, calling `df.index.dt.hour` raises an `AttributeError`. Access index attributes directly: `df.index.hour`.
- **String comparison sorting mismatches**: Slicing time series data using partial date strings (e.g. `df.loc['2026-06-01':'2026-06-05']`) raises a `KeyError` if the DatetimeIndex is not sorted chronologically. Always sort the index: `df = df.sort_index()`.
- **Confusing timezone conversions**: Running `.tz_convert('UTC')` on a timezone-naive series raises a `TypeError: Cannot convert tz-naive timestamps`. You must localize the timezone first using `.tz_localize()`.
- **Assuming `.shift()` adjusts index dates**: Running `df.shift(1)` shifts the data values down by 1 row but leaves the index dates unchanged. To shift the actual index timestamps, pass a frequency: `df.shift(1, freq='D')`.

---

## Best Practices

- Always sort the DatetimeIndex immediately after setting it to enable partial string slicing.
- Use `pd.to_datetime(..., errors='coerce')` when parsing dates from mixed-format string logs to handle errors safely.
- Use the `.dt` accessor to extract calendar properties (like day names or hours) for grouping and analysis.
- Localize naive time series immediately after loading to ensure consistent calculations across different timezones.

---

## Worked Real-World Examples

### Worked Example 1: Hourly Server CPU Load Analysis

```python
import pandas as pd

# Ingested CPU metrics
cpu_logs = pd.DataFrame({
    "Timestamp": pd.date_range(start="2026-06-01 00:00", periods=6, freq="H"),
    "CPU_Load": [45, 52, 60, 85, 90, 78]
})

# 1. Set Timestamp as the index
cpu_logs = cpu_logs.set_index("Timestamp")

# 2. Extract load differences hour-over-hour
cpu_logs["Load_Diff"] = cpu_logs["CPU_Load"].diff()

# 3. Filter high load logs during working hours (08:00 to 18:00)
work_hours_load = cpu_logs.between_time("08:00", "18:00")

print("--- CPU Load and Changes ---")
print(cpu_logs)
```

### Output

```text
--- CPU Load and Changes ---
                     CPU_Load  Load_Diff
Timestamp                               
2026-06-01 00:00:00        45        NaN
2026-06-01 01:00:00        52        7.0
2026-06-01 02:00:00        60        8.0
2026-06-01 03:00:00        85       25.0
2026-06-01 04:00:00        90        5.0
2026-06-01 05:00:00        78      -12.0
```

---

### Worked Example 2: Financial Stock Lag Calculations

```python
import pandas as pd

# Daily stock price register
stock_prices = pd.DataFrame({
    "Close": [150.0, 152.5, 149.0, 153.0],
}, index=pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]))

# Calculate 2-day historical price lag
stock_prices["Lag_2D"] = stock_prices["Close"].shift(2)

print("--- Stock Lag Matrix ---")
print(stock_prices)
```

### Output

```text
--- Stock Lag Matrix ---
            Close  Lag_2D
2026-06-01  150.0     NaN
2026-06-02  152.5     NaN
2026-06-03  149.0   150.0
2026-06-04  153.0   152.5
```

---

### Worked Example 3: Extracting Holiday Business Registers

```python
import pandas as pd

# Order date list
orders = pd.DataFrame({
    "OrderID": [1001, 1002, 1003],
    "Order_Date": pd.to_datetime(["2026-06-05", "2026-06-06", "2026-06-07"])  # Fri, Sat, Sun
})

# Keep only business day orders (Monday through Friday)
orders["Day_Num"] = orders["Order_Date"].dt.dayofweek
business_orders = orders[orders["Day_Num"] < 5]

print("--- Business Day Orders ---")
print(business_orders)
```

### Output

```text
--- Business Day Orders ---
   OrderID Order_Date  Day_Num
0     1001 2026-06-05        4
```

---

## Practice Questions

1. Explain the differences between the `Timestamp` class and the `DatetimeIndex` class in Pandas.
2. Write a command to generate a daily timeline of dates spanning the entire year of 2026.
3. How does partial string indexing behave when selecting a month (e.g. `"2026-06"`) on a DatetimeIndexed DataFrame?
4. Write a command to shift the values of a Series `s` backward by 3 periods.
5. Explain the differences between timezone localization (`.tz_localize()`) and timezone conversion (`.tz_convert()`).
6. Write a command to extract the day of the year (1 to 365) from a column of datetimes using the `.dt` accessor.
7. What error occurs when you attempt to slice a time series DataFrame that has an unsorted DatetimeIndex?
8. Write a script that shifts a DataFrame's index dates forward by 2 weeks instead of shifting the cell values.
9. How can you filter a DatetimeIndexed DataFrame to keep only rows that fall between 9:00 AM and 5:00 PM?
10. Describe how to calculate the difference in days between two columns of datetime objects.

---

## Mini Assignments

### Assignment 1: Web Traffic Lag Features
- Create a web page traffic dataset with daily page views for the month of June 2026.
- Set the date column as the index.
- Create 1-day, 2-day, and 7-day lagged page view columns to serve as features for forecasting.

### Assignment 2: Sensor Logging Timezone Standardization
- Create a sensor logging DataFrame containing UTC timestamps.
- Localize the timestamps to UTC.
- Convert the timestamps to Eastern Time (US/Eastern) and Indian Time (Asia/Kolkata).
- Extract the hour and day of the week for each timezone.

### Assignment 3: Retail Business Days Sales Audit
- Create a transaction DataFrame with dates spanning two weeks (including weekends).
- Filter out weekend sales.
- For the remaining business days, group the sales by day name (Monday, Tuesday, etc.) and calculate the average sales.

---

## Interview-Oriented Questions

- **What is the difference between standard integer shifts and index frequency shifts in Pandas?**
  - *Answer*: An integer shift (e.g. `df.shift(1)`) moves the data values down by rows but leaves the index labels unchanged, resulting in missing values at the beginning of the DataFrame. An index frequency shift (e.g. `df.shift(1, freq='D')`) shifts the index dates themselves forward, keeping the data values aligned with their original rows.
- **Why must a DatetimeIndex be sorted before using partial string slicing?**
  - *Answer*: Slicing relies on index label ranges. If the index is unsorted, Pandas cannot resolve the start and end boundaries of the range efficiently, raising a `KeyError` or returning incorrect results.
- **How does timezone localization differ from timezone conversion?**
  - *Answer*: Timezone localization (`.tz_localize()`) associates a timezone with a naive datetime object that has no timezone metadata. Timezone conversion (`.tz_convert()`) shifts an existing timezone-aware datetime object to a new timezone, adjusting the hour and date values based on the timezone offset.
- **How does the `.dt.day_name()` method extract weekday names, and how does it handle missing datetimes?**
  - *Answer*: `.dt.day_name()` extracts the weekday name (e.g. `'Monday'`) as a string based on the datetime value. If the datetime entry is `NaN` (`NaT`), the method returns `NaN`.
- **Explain the purpose of the `NaT` object in Pandas.**
  - *Answer*: `NaT` (Not a Time) is the sentinel value used by Pandas to represent missing values in datetime columns. It is the temporal equivalent of NumPy's `NaN` and propagates through datetime operations.

---

## Teaching Notes for This Chapter

- **Deconstruct Time Frequencies**: Explain frequency codes (e.g. `B` for business days, `W` for weekly, `MS` for month-start) on a calendar layout.
- **Differentiate between Series and Index datetime access**: Remind students that `.dt` is only for Series objects; indices access properties (like `.hour` or `.day`) directly.
- **Emphasize timezone safety**: Encourage students to localize naive timestamps immediately after loading to prevent errors during timezone conversions.

---

## Chapter Wrap-up Concepts Students Must Master

- Datetime Indexes enable time-based operations like partial string slicing and sorting.
- Use `pd.date_range()` to generate chronological timelines with custom frequencies.
- Use `.shift(periods)` to lag or lead data, and use frequency parameters to shift index dates.
- Use `.tz_localize()` to set a timezone and `.tz_convert()` to convert between timezones.
- Use the `.dt` accessor to extract calendar properties (like hours or day names) from datetime columns.
- `NaT` represents missing values in datetime columns.
