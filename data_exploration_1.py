# Databricks notebook source
# MAGIC %md
# MAGIC # person

# COMMAND ----------

df=spark.table("workspace.default.person")
df.count()

# COMMAND ----------

df.display()

# COMMAND ----------

df['Person Time of Death'].unique()
df['Person Time of Death'].value_counts()

# COMMAND ----------

df.columns

# COMMAND ----------



# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Load your data into a DataFrame
# df = pd.read_csv('your_data.csv')  # Uncomment and modify with your file path

# Plotting a histogram of Person Age
# Convert 'Person Age' to numeric, setting errors='coerce' will convert non-numeric values to NaN
df['Person Age'] = pd.to_numeric(df['Person Age'], errors='coerce')

# Drop NaN values from 'Person Age' for plotting
df = df.dropna(subset=['Person Age'])

plt.figure(figsize=(10, 6))
plt.hist(df['Person Age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Person Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Set x-axis ticks to be more readable
age_min = int(df['Person Age'].min())
age_max = int(df['Person Age'].max())
plt.xticks(range(age_min, age_max + 1, 5))  # This sets the ticks to every 5 years

plt.grid(True)
plt.show()


# Plotting a bar chart for Person Gender
gender_counts = df['Person Gender'].value_counts()
plt.figure(figsize=(10, 6))
gender_counts.plot(kind='bar', color='lightgreen')
plt.title('Count of Persons by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming 'df' is your DataFrame and it contains a column 'Person Blood Alcohol Content Test Result'
# that you want to plot.

# Convert the data to numeric, coercing errors to NaN (which will then be dropped)
data = pd.to_numeric(df['Person Blood Alcohol Content Test Result'], errors='coerce').dropna()

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='blue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Distribution of Blood Alcohol Content Test Results')
plt.xlabel('Blood Alcohol Content')
plt.ylabel('Frequency')
plt.grid(True)

# Customizing x-axis
ticks = np.arange(min(data), max(data), step=(max(data)-min(data))/10)  # Adjust step for more or less labels
plt.xticks(ticks, [f"{x:.2f}" for x in ticks])  # Formatting to 2 decimal places

plt.show()





# Plotting a pie chart for Ethnicity distribution
ethnicity_counts = df['Person Ethnicity'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(ethnicity_counts, labels=ethnicity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Ethnicity Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# COMMAND ----------

#nan in data
import pandas as pd

# Load your data into a DataFrame
# df = pd.read_csv('your_data.csv')  # Uncomment and modify with your file path

# Calculating NaN counts per column
nan_counts = df.isnull().sum()

# Displaying the NaN counts for each column
print(nan_counts)

# COMMAND ----------

pip install pandas-profiling


# COMMAND ----------

import pydantic_core

print("Pydantic Core Version:", pydantic_core.__version__)


# COMMAND ----------

import pandas as pd
import pandas_profiling

# Assuming df is your DataFrame
# Generate the profile report
profile = pandas_profiling.ProfileReport(df, explorative=True)

# To display the report in a Jupyter notebook
profile.to_notebook_iframe()

# To save the report as an HTML file
profile.to_file("df_profile_report.html")


# COMMAND ----------

# MAGIC %md
# MAGIC # crash

# COMMAND ----------

crash=spark.table("workspace.default.crash")


# COMMAND ----------

crash=crash.toPandas()

# COMMAND ----------

crash.display()

# COMMAND ----------

print(crash.columns)

# COMMAND ----------

#nan in data
import pandas as pd

# Load your data into a DataFrame
# df = pd.read_csv('your_data.csv')  # Uncomment and modify with your file path

# Calculating NaN counts per column
nan_counts = crash.isnull().sum()

# Displaying the NaN counts for each column
print(nan_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature importance

# COMMAND ----------

crash.info()

# COMMAND ----------

for c in crash.columns:
    print(c)

# COMMAND ----------

crash['Person Death Count'].unique()

# COMMAND ----------

import pandas as pd

# Assuming 'crash' is your DataFrame already loaded with the data

# Step 1: Filter out non-numerical columns
numerical_crash = crash.select_dtypes(include=['int64', 'float64'])

# Step 2: Compute the correlation matrix
correlation_matrix = numerical_crash.corr()

# Step 3: Extract correlations related to 'Person Death Count'
correlations_with_death_count = correlation_matrix['Person Death Count'].drop('Person Death Count', errors='ignore')  # Ignore self-correlation

# Display the correlations sorted by absolute value to see the strongest relationships, either positive or negative
print(correlations_with_death_count.abs().sort_values(ascending=False))


# COMMAND ----------

crash['Fatal Crash Flag'].unique()
