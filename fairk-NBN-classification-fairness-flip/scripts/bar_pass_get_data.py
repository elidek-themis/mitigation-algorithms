import pandas as pd

df = pd.read_csv("../data/law_dataset.csv")


unique_values_series = df.apply(lambda col: col.dropna().unique())

unique_values_series = unique_values_series.apply(lambda x: list(x))
missing_values = df.isna().sum()

columns_to_dump = ["racetxt"]
df_cleaned = df.drop(columns=columns_to_dump)

df_cleaned["male"] = df_cleaned["male"].replace({0:"female",1:"male"})
df_cleaned["pass_bar"] = df_cleaned["pass_bar"].replace({0:2,1:1})

df_cleaned.rename(columns={"male": "Sex"}, inplace=True)
df_cleaned.rename(columns={"pass_bar": "class"}, inplace=True)

df_cleaned.to_csv("../data/law_dataset_wtarget.csv")

