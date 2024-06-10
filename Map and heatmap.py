# Databricks notebook source
# MAGIC %md
# MAGIC ## Notebook that shows crashes in a map or as a heatmap

# COMMAND ----------

# MAGIC %pip install ipyleaflet==0.17.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import ipywidgets as widgets
import pandas as pd
import ipywidgets as w
from ipyleaflet import Map, Marker, MarkerCluster, Heatmap
from pyspark.sql.functions import when, col
from pyspark.sql.types import DoubleType

# Load dataset
df = spark.table("workspace.default.crash")
df = df.select("Latitude", "Longitude")

# Convert string columns to double
df = df.withColumn(
    "Latitude",
    when(
        col("Latitude").isNull(), 
        None
    ).otherwise(
        when(
            col("Latitude") == "No Data", 
            None
        ).otherwise(
            col("Latitude").cast(DoubleType())
        )
    )
)

df = df.withColumn(
    "Longitude",
    when(
        col("Longitude").isNull(), 
        None
    ).otherwise(
        when(
            col("Longitude") == "No Data", 
            None
        ).otherwise(
            col("Longitude").cast(DoubleType())
        )
    )
)

# Drop other columns
df = df.dropna(subset=["Latitude", "Longitude"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Map showing crash locations

# COMMAND ----------

pandas_df = df.toPandas() 
 
city_map = Map(center=(30.2667, -97.7333), zoom=11)
locations = [Marker(location=(y, x), draggable=False) for (x, y) in zip(pandas_df.Longitude, pandas_df.Latitude)]
cluster = MarkerCluster(markers=locations)
city_map.add(cluster)
city_map

# COMMAND ----------

# MAGIC %md
# MAGIC ### Heatmap showing crash locations

# COMMAND ----------

locations = [[y, x, 10] for (x, y) in zip(pandas_df.Longitude, pandas_df.Latitude)]
city_heatmap = Heatmap(locations=locations, radius=10)
city_map = Map(center=(30.2667, -97.7333), zoom=10)
city_map.add(city_heatmap)
city_map
