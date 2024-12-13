import os.path
import sys

import pandas as pd
from matplotlib import pyplot as plt

current = os.path.dirname((os.path.realpath(__file__)))
parent = os.path.dirname(current)
sys.path.append(parent)

csv_path =os.path.join(parent, "data/law_dataset_label_flip"
                               "/results/most_common_flip_results.csv")
df = pd.read_csv(csv_path)

plt.figure(figsize=(18, 10))
plt.plot( df['iteration'],df['number_indices_flipped'], label='Sensitive Attributes Predicted)', color='blue')

for i, val in enumerate(df['sum_indices_flipped']):
    plt.scatter(df['iteration'][i], df['number_indices_flipped'][i], color='orange', zorder=4, label='Negative to Positive Flip (protected class)' if i == 0 else "")
    plt.text(df['iteration'][i], df['number_indices_flipped'][i], str(val), color='black', fontsize=8, ha='center', va='bottom')


plt.xlim(0, max(df['iteration']))
plt.ylim(0, max(df['number_indices_flipped']) + 5)  # St

plt.xticks(df['iteration'], fontsize=10, rotation=45, ha='right')
n = max(1, len(sorted(df['number_indices_flipped'])) //20 )
plt.yticks(df['number_indices_flipped'][::n], fontsize=12, rotation=45, ha='right')

plt.title('Most Common Flip Results')
plt.xlabel('Number of neighbors (from train set) flipped from negative to positive')
plt.ylabel('Negative classified from the sensitive-class flipped positive')
plt.legend()
plt.grid()
plt.tight_layout(pad=0.5)
plt.savefig('law_flip_most_common_label_flip.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)
