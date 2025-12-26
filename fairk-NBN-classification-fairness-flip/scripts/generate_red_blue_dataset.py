import pandas as pd
import numpy as np


def generate_dataset(n_rows=1000, red_negative_ratio=1):
    data = {
        "color": [],
        "value": [],
        "value1":[],
        "class": []
    }

    for _ in range(n_rows):
        color = np.random.choice(["red", "blue"])

        if color == "blue":
            class_label = np.random.choice([1, 2], p=[0.7,0.3])
        else:
            class_label = np.random.choice([1, 2], p=[1 - red_negative_ratio, red_negative_ratio])

        if color == "red":
            if class_label == 1:
                value = np.random.normal(5, 5)
                value1 = np.random.normal(5, 5)
            else:
                value1 = np.random.normal(-9, 15)
                value = np.random.normal(-9, 15)
        else:  # Blue
            if class_label == 1:
                value1 = np.random.normal(0, 2)
                value = np.random.normal(0, 2)
            else:
                value1 = np.random.normal(3, 5)

                value = np.random.normal(3, 5)

        data["color"].append(color)
        data["value"].append(value)
        data["value1"].append(value1)
        data["class"].append(class_label)

    df = pd.DataFrame(data)
    return df


dataset = generate_dataset(n_rows=2000, red_negative_ratio=0.8)
dataset.to_csv("red_blue_dataset.csv", index=False)