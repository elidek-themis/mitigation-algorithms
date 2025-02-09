import logging
import os
import time
import hydra
import numpy as np
import matplotlib.lines as mlines
import imageio
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv
from src.config.schema import Config
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.jobs.fairness_parity import FairnessParity
from src.utils.preprocess_utils import flip_value

load_dotenv()
hydra_config_path = '../' +os.getenv("HYDRA_CONFIG_PATH")
hydra_config_name = os.getenv("HYDRA_CONFIG_NAME")

def merge_t_dfs(df1, df2):
    df1 =df1.copy()
    df2 = df2.copy()

    df1['color'] = 1
    df2['color'] = 0

    selected_train_df = pd.concat([df1, df2], ignore_index=False, sort=False)
    return selected_train_df

def make_plot(fp,selected_val_df,neighbor_train_df, num=None):

    x_min, x_max = fp.x_train['value'].min(), fp.x_train['value'].max()
    y_min, y_max = fp.x_train['value1'].min(), fp.x_train['value1'].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))

    mesh_df = pd.DataFrame({'value': xx.ravel(), 'value1': yy.ravel()})
    mesh_df['prediction'] = fp.model.predict(mesh_df[['value', 'value1']])
    Z = mesh_df['prediction'].values.reshape(xx.shape)

    plt.figure(figsize=(30, 15))
    bright_cmap = mcolors.ListedColormap(['#FFCC00', '#339933'])
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=bright_cmap)


    train_colors = ['red' if cls == 1 else 'blue' for cls in neighbor_train_df['color']]
    train_scatter =plt.scatter(neighbor_train_df['value'], neighbor_train_df['value1'],
               c=train_colors, marker='o', s=120, edgecolor='k', label='Train Neighbors', zorder=1)

    val_colors = ['red' if cls == 1 else 'blue' for cls in selected_val_df['color']]
    val_scatter =plt.scatter(selected_val_df['value'], selected_val_df['value1'],
                c=val_colors, marker='x',s=80, label='Validation Points', zorder=2)

    filename = f"frame_{num}.png"

    yellow_patch = mlines.Line2D([], [], color='#FFCC00', lw=4, label='Positive')
    green_patch = mlines.Line2D([], [], color='#339933', lw=4, label='Negative')

    plt.legend(handles=[yellow_patch, green_patch, train_scatter, val_scatter], loc='upper left', fontsize=16)
    plt.title('Decision Boundary')
    plt.savefig(filename, dpi=500, bbox_inches='tight', pad_inches=0.04)
    print(f"Creating Frame{num+1}")

    plt.close()

    return filename

def make_gif(frames):
    with imageio.get_writer("fair_knn.gif", mode="I", duration=4, loop=100) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)




def get_train_representation(fp):
    train_indexes = list(fp.reverse_index.keys())
    neighbor_train = fp.x_train.loc[train_indexes]
    sensitive_val = [fp.y_train_sensitive_attr[i] for i, idx in enumerate(fp.x_train.index) if idx in train_indexes]
    neighbor_train['color'] = sensitive_val
    return neighbor_train

@hydra.main(config_path=hydra_config_path, config_name=hydra_config_name, version_base=None)
def main(config: Config):
    frames = []
    fp = FairnessParity(config)
    reslts_df, train_indexer = fp.run_fairness_parity()
    neighbor_train_df = get_train_representation(fp)
    selected_val_df = merge_t_dfs(fp.t0, fp.t1)
    frame = make_plot(fp,selected_val_df,neighbor_train_df,num=0)
    frames.append(frame)
    for i,index in enumerate(train_indexer):
        fp.y_train = flip_value(fp.y_train, index, fp.class_positive_value,
                                  fp.config.data.class_attribute.name)
        fp._train()
        frame =make_plot(fp,selected_val_df,neighbor_train_df,num=i)
        frames.append(frame)

    make_gif(frames)

if __name__ == "__main__":
    main()
