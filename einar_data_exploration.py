# Databricks notebook source


# COMMAND ----------

df = spark.table("workspace.default.person")

display(df)
