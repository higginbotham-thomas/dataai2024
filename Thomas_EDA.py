# Databricks notebook source
# MAGIC %pip install --upgrade pandas

# COMMAND ----------

# MAGIC %pip install --upgrade ydata_profiling

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import ydata_profiling

# COMMAND ----------

df=spark.table("workspace.default.crash")
df.count()

# COMMAND ----------

type(df)

# COMMAND ----------

df.head()

# COMMAND ----------

import pandas as pd
from ydata_profiling import ProfileReport

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

# Generate the profile report
report = ProfileReport(pandas_df,
                       title='Crash EDA',
                       infer_dtypes=False,
                       interactions=None,
                       missing_diagrams=None,
                       correlations={"auto": {"calculate": False},
                                     "pearson": {"calculate": True},
                                     "spearman": {"calculate": True}})

# COMMAND ----------

pandas_df.display()

# COMMAND ----------

# Print data types of all columns
print("Data types of all columns in the DataFrame:")
#print(pandas_df.dtypes)

# Optionally, format the output more nicely
data_types = pandas_df.dtypes
for column, dtype in data_types.items():
    print(f"{column}: {dtype}")

# COMMAND ----------

# Convert date columns to datetime and handle nonzero nanoseconds
for column in pandas_df.columns:
    if pandas_df[column].dtype == 'object':
        try:
            pandas_df[column] = pd.to_datetime(pandas_df[column], errors='coerce')
        except Exception as e:
            print(f"Error converting column {column} to datetime: {e}")


# COMMAND ----------

# Generate the profile report
report = ProfileReport(pandas_df,
                       title='Crash EDA',
                       infer_dtypes=False,
                       interactions=None,
                       missing_diagrams=None,
                       correlations={"auto": {"calculate": False},
                                     "pearson": {"calculate": True},
                                     "spearman": {"calculate": True}})

# COMMAND ----------

import dateutil

# COMMAND ----------

# Save the profile report to an HTML file
#report.to_file("/dbfs/FileStore/profile_report.html")

# Display the profile report in Databricks notebook
#displayHTML(profile.to_html())

# Optionally, display the DataFrame as HTML
report_html = report.to_html()
displayHTML(report_html)



# COMMAND ----------

pd.Timestamp
