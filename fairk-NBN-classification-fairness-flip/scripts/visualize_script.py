import os.path
import sys

import pandas as pd
from matplotlib import pyplot as plt

current = os.path.dirname((os.path.realpath(__file__)))
parent = os.path.dirname(current)
sys.path.append(parent)

csv_path =os.path.join(parent, "data/red_blue_ida"
                               "/results/most_common_flip_results.csv")
df = pd.read_csv(csv_path)

plt.figure(figsize=(18, 10))
plt.plot( df['train_val_flipped'],df['number_flipped'], label='Validation Data', color='blue')

scatter_interval = 1
for i in range(0, len(df['train_val_flipped']), scatter_interval):
    val_sa = df['sum_sa_indices_flipped'][i]
    val_dom = df['sum_indices_flipped'][i] - df['sum_sa_indices_flipped'][i]
    plt.scatter(df['train_val_flipped'][i], df['number_flipped'][i], color='orange', zorder=4, label='Negative to Positive Flip' if i == 0 else "")
    plt.text(df['train_val_flipped'][i], df['number_flipped'][i],
             f'S:{val_sa}\nD:{val_dom}', color='black', fontsize=8,
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=1))


plt.xlim(0, max(df['train_val_flipped']))
plt.ylim(0, max(df['number_flipped']) + 5)


x_tick_interval = 1
x_ticks = df['train_val_flipped'][::x_tick_interval]
plt.xticks(x_ticks, fontsize=10, rotation=45, ha='right')
n = max(1, len(sorted(df['number_flipped'])) //20 )
plt.yticks(df['number_flipped'][::n], fontsize=12, rotation=45, ha='right')

plt.title('Most Common Flip Results')
plt.xlabel('Number of neighbors (from train set) flipped from negative to positive')
plt.ylabel('Negative classified validation points flipped positive')
plt.legend()
plt.grid()
plt.tight_layout(pad=0.5)
plt.savefig('diabetic_ida_flip_most_common_label_flip.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)
