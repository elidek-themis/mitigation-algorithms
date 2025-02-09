import os.path
import sys

import pandas as pd
from matplotlib import pyplot as plt

current = os.path.dirname((os.path.realpath(__file__)))
parent = os.path.dirname(current)
sys.path.append(parent)

csv_path =os.path.join(parent, "data/law_ida"
                               "/results/most_common_flip_results.csv")
df = pd.read_csv(csv_path)

def visualize_dom_attr_vs_sen_attr(df,name):
    plt.figure(figsize=(10, 6))

    min_dom = df['number_dom_attr_predicted_positive'].min()
    plt.axhline(y=min_dom, color='red', linestyle='--', linewidth=1.5, label=f'Flip Goal of Sensitive Attr Predicted Positive')

    plt.plot(df['train_val_flipped'], df['number_sensitive_attr_predicted_positive'], label='Sensitive Attr Predicted Positive')
    plt.plot(df['train_val_flipped'], df['number_dom_attr_predicted_positive'], label='Dominant Attr Predicted Positive')
    plt.xlabel('train_val_flipped')
    plt.ylabel('Predicted Positive Count')
    plt.title('Comparison of Sensitive vs Dominant Attribute Predictions Over Iterations')
    plt.legend()
    plt.savefig(f'{name}.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)


visualize_dom_attr_vs_sen_attr(df, 'diabetic_ida_dom_attrVsen_attr')