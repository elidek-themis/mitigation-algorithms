import os.path
import sys

import pandas as pd
from matplotlib import pyplot as plt

current = os.path.dirname((os.path.realpath(__file__)))
parent = os.path.dirname(current)
sys.path.append(parent)

csv_path =os.path.join(parent, "data/law_dataset_label_drop"
                               "/results/label_drop_attempt1.csv")
df = pd.read_csv(csv_path)

'''
filtered_df = df[df['iteration'] % 3 == 0]
first_row = df.iloc[[0]]
# Last row
last_row = df.iloc[[-1]]

# Combine the filtered DataFrame with the first and last rows, then drop duplicates
final_df = pd.concat([filtered_df, first_row, last_row]).drop_duplicates().reset_index(drop=True)
'''
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['number_sensitive_attr_predicted_positive'], label='Sensitive Attr Predicted Positive')
plt.plot(df['iteration'], df['number_dom_attr_predicted_positive'], label='Dominant Attr Predicted Positive')
plt.xlabel('Iteration')
plt.ylabel('Predicted Positive Count')
plt.title('Comparison of Sensitive vs Dominant Attribute Predictions Over Iterations')
plt.legend()
plt.savefig('diabetic_label_drop_sensitiveVSpositive.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)



plt.figure(figsize=(10, 6))

plt.plot( df['iteration'],df['pos_t0'], label='Sensitive Attributes Predicted)', color='blue')
for i in range(0, len(df['sum_indices_removed']), 2):
    # Scatter plot for each point, only label the first point
    plt.scatter(df['iteration'][i], df['pos_t0'][i], color='orange', zorder=4,
                label='Sum of Label Drops' if i == 0 else "")

    # Add text annotation for each point
    plt.text(df['iteration'][i], df['pos_t0'][i], str(df['sum_indices_removed'][i]),
             color='black', fontsize=8, ha='center', va='bottom')
plt.xlim(0, max(df['iteration']))
plt.ylim(0, max(df['pos_t0']) + 5)  # St

plt.xticks(df['iteration'], fontsize=10, rotation=45, ha='right')
n = max(1, len(sorted(df['pos_t0'])) //10 )
h = max(1, len(sorted(df['iteration'])) //20 )

plt.yticks(df['pos_t0'][::n], fontsize=12, rotation=45, ha='right')
plt.xticks(df['iteration'][::h], fontsize=12, rotation=45, ha='right')

plt.xlabel('Iteration')
plt.ylabel('Negative-to-Positive Change on the Negative Classified val set of the Protected Class)')
plt.title('Label Change Over Iterations')
plt.legend()
plt.grid()
plt.show()
plt.tight_layout(pad=0.5)
plt.legend()
plt.savefig('diabetic_label_drop_sensitiveVSpositive_.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)





