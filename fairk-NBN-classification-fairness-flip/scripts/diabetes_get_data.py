import numpy as np
import pandas as pd
def convert_to_midpoint(age_range):
    lower, upper = map(int, age_range.strip('[)').split('-'))
    return (lower + upper) / 2
df = pd.read_csv("../data/diabetic_data.csv")

columns_to_dump = ["race", "encounter_id", "patient_nbr", "weight", "payer_code","max_glu_serum","A1Cresult","medical_specialty"]

rows_to_drop = (
    (df['admission_type_id'] == 6) |
    (df['admission_type_id'] == 5) |
    (df['discharge_disposition_id'] == 18) |
    (df['discharge_disposition_id'] == 25) |
    (df['discharge_disposition_id'] == 26) |
    (df['diag_1'] == '?') |
    (df['diag_2'] == '?') |
    (df['diag_3'] == '?') |
    (df['gender'] == 'Unknown/Invalid') |
    (df['medical_specialty'].isna())
)



df_cleaned = df.loc[~rows_to_drop].drop(columns=columns_to_dump)
question_mark_counts = (df_cleaned == "?").sum()
missing_values = df_cleaned.isna().sum()
df_cleaned['age'] = df_cleaned['age'].apply(convert_to_midpoint)


cols_4_values = [
    col for col in df_cleaned.columns
    if set(df_cleaned[col].dropna().unique()) == {"Up", "Down", "Steady", "No"}
]
cols_4_values.extend(["acetohexamide","tolbutamide","troglitazone","tolazamide","citoglipton","examide","glipizide-metformin","glimepiride-pioglitazone",
                      "metformin-rosiglitazone","metformin-pioglitazone"])

mappings_4_values = {"Up": 1, "Down": -2, "Steady": 0, "No": -1}
df_cleaned[cols_4_values] = df_cleaned[cols_4_values].replace(mappings_4_values)

cols_yes_no = [
    col for col in df_cleaned.columns
    if set(df_cleaned[col].dropna().unique()) == {"Yes", "No"}
]
mappings_yes_no = {"Yes": 1, "No": 0}
df_cleaned[cols_yes_no] = df_cleaned[cols_yes_no].replace(mappings_yes_no)

df_cleaned["change"] = df_cleaned["change"].replace({"Ch":1,'No':0})
df_cleaned["readmitted"] = df_cleaned["readmitted"].replace({">30":2,"<30":2,'NO':1})
df_cleaned["gender"] = df_cleaned["gender"].replace({"Female":"female","Male":"male"})


unique_values_series = df_cleaned.apply(lambda col: col.dropna().unique())

unique_values_series = unique_values_series.apply(lambda x: list(x))

df_cleaned.rename(columns={"readmitted": "class"}, inplace=True)
df_cleaned.rename(columns={"gender": "Sex"}, inplace=True)


df_cleaned.to_csv("../data/diabetic_data_wtarget.csv")
