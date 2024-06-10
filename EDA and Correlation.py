# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------


import pandas as pd
import dateutil
import ydata_profiling
from ydata_profiling import ProfileReport

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

df=spark.table("workspace.default.crash")
df.count()

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

# COMMAND ----------

columns_to_include = [
    'Vehicle Make', 'Vehicle Model Name', 'Vehicle Model Year', 
    'Vehicle Parked Flag',
    'Person Age', 'Person Ethnicity', 'Person Gender', 
    'Crash Date', 'Crash Time'
]

# Filter the DataFrame to include only the specified columns
pandas_df = pandas_df.filter(items=columns_to_include)

# Display the filtered DataFrame
print("Filtered DataFrame with specified columns:")
print(pandas_df)

# Optionally, display the filtered DataFrame as HTML in Databricks
pandas_df_html = pandas_df.to_html()
#displayHTML(filtered_df_html)

# COMMAND ----------



# Convert categorical columns to numeric
label_encoders = {}
for column in pandas_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    pandas_df[column] = le.fit_transform(pandas_df[column].astype(str))
    label_encoders[column] = le



# Convert Crash Date to datetime
pandas_df['Crash Date'] = pd.to_datetime(pandas_df['Crash Date'], errors='coerce')

# Extract day of week and month number
pandas_df['Crash Day of Week'] = pandas_df['Crash Date'].dt.dayofweek  # Monday=0, Sunday=6
pandas_df['Crash Month'] = pandas_df['Crash Date'].dt.month

# Convert Crash Time from numeric to time
def convert_to_time(crash_time):
    if pd.isna(crash_time):
        return pd.NaT
    crash_time_str = f'{int(crash_time):04}'
    return pd.to_datetime(crash_time_str, format='%H%M').time()

pandas_df['Crash Time'] = pandas_df['Crash Time'].apply(convert_to_time)

# Create a new column to group times into categories
def categorize_time(crash_time):
    if pd.isna(crash_time):
        return 'Unknown'
    if crash_time >= pd.to_datetime('00:00').time() and crash_time < pd.to_datetime('06:00').time():
        return '1'
    elif crash_time >= pd.to_datetime('06:00').time() and crash_time < pd.to_datetime('12:00').time():
        return '2'
    elif crash_time >= pd.to_datetime('12:00').time() and crash_time < pd.to_datetime('18:00').time():
        return '3'
    elif crash_time >= pd.to_datetime('18:00').time() and crash_time <= pd.to_datetime('23:59').time():
        return '4'

pandas_df['Crash Time Category'] = pandas_df['Crash Time'].apply(categorize_time)

# Drop the original Crash Date and Crash Time columns
pandas_df = pandas_df.drop(columns=['Crash Date', 'Crash Time'])


# COMMAND ----------

# Converting the spark df to a pandas df
pandas_df = pandas_df.dropna(axis=1, how='all')


# COMMAND ----------

pandas_df

# COMMAND ----------


# Compute the correlation matrix
correlation_matrix = pandas_df.corr()

# Unstack the matrix to a long format
correlation_pairs = correlation_matrix.unstack()

# Convert to a DataFrame and reset index
correlation_pairs_df = pd.DataFrame(correlation_pairs, columns=['Correlation']).reset_index()

# Rename columns for clarity
correlation_pairs_df.columns = ['Variable1', 'Variable2', 'Correlation']

# Filter to include only correlations where Variable1 is 'Fatal Crash Flag'
correlation_pairs_df = correlation_pairs_df[correlation_pairs_df['Variable1'] == 'Fatal Crash Flag']

# Exclude any values where Variable2 contains the words "death" or "injured"
correlation_pairs_df = correlation_pairs_df[
    ~correlation_pairs_df['Variable2'].str.contains(r'death|injured|Injury', case=False, na=False)
]

# Sort by absolute correlation values, but keep original values for context
correlation_pairs_df['AbsCorrelation'] = correlation_pairs_df['Correlation'].abs()
correlation_pairs_df = correlation_pairs_df.sort_values(by='AbsCorrelation', ascending=False)

# Filter out self-correlations (where Variable1 == Variable2)
correlation_pairs_df = correlation_pairs_df[correlation_pairs_df['Variable1'] != correlation_pairs_df['Variable2']]

# Drop the auxiliary column
correlation_pairs_df = correlation_pairs_df.drop(columns=['AbsCorrelation'])

# Filter out correlations greater than 0.9 or less than -0.9
#correlation_pairs_df = correlation_pairs_df[(correlation_pairs_df['Correlation'] <= 0.9) & 
#                                            (correlation_pairs_df['Correlation'] >= -0.9)
#]

# Display the sorted correlation pairs
#print("Sorted Correlation Pairs:")
#print(correlation_pairs_df)

# Optionally, display the sorted correlation pairs as HTML
correlation_pairs_html = correlation_pairs_df.to_html()
displayHTML(correlation_pairs_html)

# COMMAND ----------

# Generate the profile report
report = ProfileReport(pandas_df,
                title='Crash Data EDA',
                infer_dtypes=False,
                interactions=None,
                missing_diagrams=None,
                correlations={"auto": {"calculate": False},
                              "pearson": {"calculate": True},
                              "spearman": {"calculate": True}})

# COMMAND ----------

pandas_df.filter(like='Date', axis=1)


# COMMAND ----------


# Optionally, visualize the correlation matrix using a heatmap
plt.figure(figsize=(50, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# COMMAND ----------

report_html = report.to_html()
displayHTML(report_html)

# COMMAND ----------

correlation_pairs_df.
