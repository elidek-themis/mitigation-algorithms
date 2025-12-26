from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
from collections import defaultdict



def save_dataset_pos_rates(df, class_name, sensitive_feature_name, dominant_value, class_positive_value, save_folder_path):

    counts_df = df.groupby([class_name, sensitive_feature_name]).size().unstack()
    sens_feature_pos_rates = counts_df.div(counts_df.sum(axis=0), axis=1)

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    with open(os.path.join(save_folder_path, "dominant_positive_rate.txt"), 'w') as f_1, \
        open(os.path.join(save_folder_path, "sensitive_positive_rate.txt"), 'w') as f_2:
        for column in sens_feature_pos_rates.columns:
            if column == dominant_value:
                f_1.write(str(sens_feature_pos_rates.loc[class_positive_value, column]) + "\n")
            else:
                f_2.write(str(sens_feature_pos_rates.loc[class_positive_value, column]) + "\n")

def encode_dataframe(df, sensitive_attr, dominant_class_value, sensitive_class_value, class_attr, class_negative_value, class_positive_value):

    encoder = LabelEncoder()
    for column in df.select_dtypes(exclude="number").columns:
        df[column] = encoder.fit_transform(df[column])
        print(f"\nEncoded column: {column} with mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

        if column == sensitive_attr:
            dominant_class_value = encoder.transform([dominant_class_value])[0]
            sensitive_class_value = encoder.transform([sensitive_class_value])[0]

        if column == class_attr:
            class_negative_value = encoder.transform([class_negative_value])[0]
            class_positive_value = encoder.transform([class_positive_value])[0]

    return dominant_class_value, sensitive_class_value, class_negative_value, class_positive_value

def save_results_with_same_name_from_folder_in_first_num_of_file_name_order(folder, name_value_delimiter, save_folder_path, save_every = 1, only_div_by = 1, num_smaller_than = 100000, num_greater_than = 0):
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path) 

    files = ([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    first_nums_of_file_names = [float(name.split('_')[0]) for name in files]
    sort_idx = np.argsort(first_nums_of_file_names)
    first_nums_of_file_names = np.array(first_nums_of_file_names)[sort_idx]
    files = np.array(files)[sort_idx]

    results_to_save_dict = defaultdict(list)
    x_values = []
    for i in range(0, len(files), save_every):
        num = first_nums_of_file_names[i]
        if (num < num_smaller_than) and (num > num_greater_than) and (not only_div_by or (num % only_div_by == 0)):
            x_values.append(first_nums_of_file_names[i])
            with open(os.path.join(folder, files[i]), 'r') as f:
                for line in f:
                    if line.strip():
                        name, value = line.split(name_value_delimiter, 1)
                        results_to_save_dict[name].append(float(value.strip()))

    with open(os.path.join(save_folder_path, "x_values.txt"), 'w') as f:
        for x in x_values:
            f.write(str(x) + "\n")

    for key, values in results_to_save_dict.items():
        if values:
            values = np.array(values)

            with open(os.path.join(save_folder_path, key + ".txt"), 'w') as f:
                for value in values:
                    f.write(str(value) + "\n")