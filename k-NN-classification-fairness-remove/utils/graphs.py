import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from utils.utilities import save_results_with_same_name_from_folder_in_first_num_of_file_name_order
import numpy as np



def plot_dataset_stats(df, class_name, sensitive_feature_name, save_folder = None, name = "default", show_plot = False):

    counts_df = df.groupby([class_name, sensitive_feature_name]).size().unstack()
    sens_feature_pos_rates = counts_df.div(counts_df.sum(axis=0), axis=1)
    ax = sens_feature_pos_rates.plot(kind="bar")
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.title("Positive and Negative Rates for each Attribute")
    plt.ylabel("Rate")
    plt.tight_layout()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, name+"_barplot.png"))

    if show_plot:
        plt.show()

    plt.close()


    first_labels = counts_df.columns.tolist()
    first_sizes = counts_df.sum(axis=0).tolist()

    second_labels = []
    second_sizes = []
    for first_label in first_labels:
        for second_label in counts_df.index.tolist():
            second_labels.append(second_label)
            second_sizes.append(counts_df[first_label][second_label])
    first_labels = [str(first_labels[i]) + str(f"\n{round(first_sizes[i]/sum(first_sizes), 5):.2%}") for i in range(len(first_labels))]
    second_labels = [str(second_labels[i]) + str(f"\n{round(second_sizes[i]/sum(second_sizes), 5):.2%}") for i in range(len(second_labels))]

    fig, ax = plt.subplots()
    first_wedges, _ = ax.pie(first_sizes, radius=2, startangle=90)
    second_wedges, _ = ax.pie(second_sizes, radius=1.4, startangle=90)

    add_rotated_labels(ax, first_wedges, first_labels, radius=1.7)
    add_rotated_labels(ax, second_wedges, second_labels, radius=1.1)

    centre_circle = plt.Circle((0, 0), 0.8, fc='white')
    fig.gca().add_artist(centre_circle)
    plt.tight_layout()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, name+".png"), transparent=True, dpi=300)

    if show_plot:
        plt.show()

    plt.close()

def add_rotated_labels(ax, wedges, labels, radius):
    for wedge, label in zip(wedges, labels):
        theta = (wedge.theta2 + wedge.theta1) / 2.0
        angle_rad = np.deg2rad(theta)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        rotation = theta - 90
        if y < 0:
            rotation += 180

        ax.text(
            x, y, label,
            ha='center', va='center',
            rotation=rotation,
            rotation_mode='anchor',
            fontsize=8)

def plot_results_from_folders(folders_to_plot,
                              x_values_file,
                              xlabel = "Iteration",
                              ylabel = "Value",
                              save_folder = None,
                              show_plot = False):
    
    x_values = []
    if x_values_file:
        with open(x_values_file, 'r') as f:
            x_values = [float(line.strip()) for line in f if line.strip()]

    for folder in folders_to_plot:
        for file in [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]:
            
            if os.path.basename(x_values_file) == file:
                continue
            
            with open(os.path.join(folder, file), 'r') as f:
                results = [float(line.strip()) for line in f if line.strip()]

                if x_values:
                    plt.plot(x_values, results, marker='o', linestyle='-')
                else:
                    plt.plot(results, marker='o', linestyle='-')
                plt.title(file)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.grid(True)
                plt.tight_layout()

                if save_folder:
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    plt.savefig(os.path.join(save_folder, os.path.splitext(file)[0]))

                if show_plot:
                    plt.show()

                plt.close()

def plot_results_together_from_files(files,
                                     add_y_lines_instead_from_files_line,
                                     x_values_file,
                                     title,
                                     x_label = "Iteration",
                                     y_label = "Value",
                                     legend_labels = None,
                                     add_lines_at_y = None,
                                     added_lines_labels = None,
                                     save_folder = None,
                                     show_plot = False):
    
    x_values = []
    if x_values_file:
        with open(x_values_file, 'r') as f:
            x_values = [float(line.strip()) for line in f if line.strip()]

    colors=["midnightblue", "darkorange", "forestgreen", "lightskyblue", "navajowhite", "lightgreen"]
    for i in range(len(files)):
        with open(files[i], 'r') as f:
            if add_y_lines_instead_from_files_line[i] == None:
                results = [float(line.strip()) for line in f if line.strip()]
            else:
                results = [float(line.strip()) for (j, line) in enumerate(f) if (line.strip() and j==add_y_lines_instead_from_files_line[i])]
                results = results * len(x_values)

            if x_values:
                if legend_labels:
                    plt.plot(x_values, results, marker='o', linestyle='-', label = legend_labels[i], color = colors[i])
                    plt.legend()
                else:
                    plt.plot(x_values, results, marker='o', linestyle='-', color = colors[i])
            else:
                if legend_labels:
                    plt.plot(results, marker='o', linestyle='-', label = legend_labels[i], color = colors[i])
                    plt.legend()
                else:
                    plt.plot(results, marker='o', linestyle='-', color = colors[i])
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True)
            plt.tight_layout()
 
    for i in range(len(add_lines_at_y)):
        plt.axhline(y=add_lines_at_y[i], linestyle='--', label=added_lines_labels[i], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        plt.legend()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, title))
    
    if show_plot:
        plt.show()

    plt.close()

def plot_results_with_same_name_from_folder_in_first_num_of_file_name_order(folder_to_plot,
                                                                            name_value_delimiter,
                                                                            save_every = 1,
                                                                            only_div_by = 1,
                                                                            num_smaller_than = 100000,
                                                                            num_greater_than = 0,
                                                                            xlabel = "K value",
                                                                            title = "Stratified 10 fold average values over k",
                                                                            save_folder = None,
                                                                            show_plot = False):
    
    files = ([f for f in os.listdir(folder_to_plot) if os.path.isfile(os.path.join(folder_to_plot, f))])
    first_nums_of_file_names = [name.split('_')[0] for name in files]
    results_to_plot_dict = defaultdict(list)
    x_values = []
    for i in range(0, len(files), save_every):
        num = float(first_nums_of_file_names[i])
        if (num < num_smaller_than) and (num > num_greater_than) and (not only_div_by or (num % only_div_by == 0)):
            x_values.append(float(first_nums_of_file_names[i]))
            with open(os.path.join(folder_to_plot, files[i]), 'r') as f:
                for line in f:
                    if line.strip():
                        name, value = line.split(name_value_delimiter, 1)
                        results_to_plot_dict[name].append(float(value.strip()))

    sort_idx = np.argsort(x_values)
    x_values = np.array(x_values)[sort_idx]
    for key, values in results_to_plot_dict.items():
        if values:
            values = np.array(values)[sort_idx]

            plt.plot(x_values, values, marker='o', linestyle='-')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(key)
            plt.grid(True)
            plt.tight_layout()

            if save_folder:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                plt.savefig(os.path.join(save_folder, title))            

            if show_plot:
                plt.show()

            plt.close()

def plot_difference_of_files(files_to_dif,
                             add_y_lines_instead_from_files_line,
                             x_values_file,
                             title,
                             x_label = "Iteration",
                             y_label = "Value",
                             legend_labels=None,
                             add_lines_at_y = None,
                             added_lines_labels = None,
                             save_folder = None,
                             show_plot = False):
    
    x_values = []
    if x_values_file:
        with open(x_values_file, 'r') as f:
            x_values = [float(line.strip()) for line in f if line.strip()]

    colors=["midnightblue", "darkorange", "forestgreen", "lightskyblue", "navajowhite", "lightgreen"]
    for i, (file_1, file_2) in enumerate(files_to_dif):
        with open(file_1, 'r') as f, \
            open(file_2, 'r') as f2:
            if add_y_lines_instead_from_files_line[i][0] == None:
                results_1 = [float(line.strip()) for line in f if line.strip()]
            else:
                results_1 = [float(line.strip()) for (j, line) in enumerate(f) if (line.strip() and j==add_y_lines_instead_from_files_line[i][0])]
                results_1 = results_1 * len(x_values)
            
            if add_y_lines_instead_from_files_line[i][1] == None:
                results_2 = [float(line.strip()) for line in f2 if line.strip()]
            else:
                results_2 = [float(line.strip()) for (j, line) in enumerate(f2) if (line.strip() and j==add_y_lines_instead_from_files_line[i][1])]
                results_2 = results_2 * len(x_values)
            
            results = [(x1 - x2) for x1, x2 in zip(results_1, results_2)]

            if x_values:
                if legend_labels:
                    plt.plot(x_values, results, marker='o', linestyle='-', label = legend_labels[i], color = colors[i])
                    plt.legend()
                else:
                    plt.plot(x_values, results, marker='o', linestyle='-', color = colors[i])
            else:
                if legend_labels:
                    plt.plot(results, marker='o', linestyle='-', label = legend_labels[i], color = colors[i])
                    plt.legend()
                else:
                    plt.plot(results, marker='o', linestyle='-', color = colors[i])
    
    for j in range(len(add_lines_at_y)):
        plt.axhline(y=add_lines_at_y[j], linestyle='--', label=added_lines_labels[j], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        plt.legend()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, title.replace("\n", "")))
        
    if show_plot:
        plt.show()

    plt.close()

def plot_f1_of_files(prec_files,
                    rec_files,
                    add_prec_y_lines_instead_from_files_line,
                    add_rec_y_lines_instead_from_files_line,
                    x_values_file,
                    title,
                    x_label = "Iteration",
                    y_label = "Value",
                    legend_labels=None,
                    add_lines_at_y = None,
                    added_lines_labels = None,
                    save_folder = None,
                    show_plot = False):
    
    x_values = []
    if x_values_file:
        with open(x_values_file, 'r') as prec_f:
            x_values = [float(line.strip()) for line in prec_f if line.strip()]

    colors=["midnightblue", "darkorange", "forestgreen", "lightskyblue", "navajowhite", "lightgreen"]
    for i, (prec_file, rec_file) in enumerate(zip(prec_files, rec_files)):
        with open(prec_file, 'r') as prec_f, \
            open(rec_file, 'r') as rec_f:
            if add_prec_y_lines_instead_from_files_line[i] == None:
                precs = [float(line.strip()) for line in prec_f if line.strip()]
            else:
                precs = [float(line.strip()) for (j, line) in enumerate(prec_f) if (line.strip() and j==add_prec_y_lines_instead_from_files_line[i])]
                precs = precs * len(x_values)
            
            if add_rec_y_lines_instead_from_files_line[i] == None:
                recs = [float(line.strip()) for line in rec_f if line.strip()]
            else:
                recs = [float(line.strip()) for (j, line) in enumerate(rec_f) if (line.strip() and j==add_rec_y_lines_instead_from_files_line[i])]
                recs = recs * len(x_values)
            
            results = [((2 * prec * rec)/(prec + rec)) for prec, rec in zip(precs, recs)]

            if x_values:
                if legend_labels:
                    plt.plot(x_values, results, marker='o', linestyle='-', label = legend_labels[i], color = colors[i])
                    plt.legend()
                else:
                    plt.plot(x_values, results, marker='o', linestyle='-', color = colors[i])
            else:
                if legend_labels:
                    plt.plot(results, marker='o', linestyle='-', label = legend_labels[i], color = colors[i])
                    plt.legend()
                else:
                    plt.plot(results, marker='o', linestyle='-', color = colors[i])
    
    for j in range(len(add_lines_at_y)):
        plt.axhline(y=add_lines_at_y[j], linestyle='--', label=added_lines_labels[j], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        plt.legend()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, title.replace("\n", "")))
        
    if show_plot:
        plt.show()

    plt.close()

def save_plots_for_dataset(data_folder, experiment_folder, x_label, save_every, num_smaller_than, only_div_by=None, num_greater_than=0, compare_name=False, add_y_line_from_comparison_over_var_file_line=None):

    test_path = os.path.join(data_folder, "test_results/10_fold_scv/", experiment_folder)
    test_save_path = os.path.join(data_folder, "test_results/10_fold_scv_over_var/", experiment_folder)
    val_path = os.path.join(data_folder, "val_results/10_fold_scv/", experiment_folder)
    val_save_path = os.path.join(data_folder, "val_results/10_fold_scv_over_var/", experiment_folder)
    plots_folder = os.path.join(data_folder, "plots/10_fold_scv_over_var/", experiment_folder + (("_vs_" + (str.lower(compare_name).replace(" ", "_"))) if compare_name else ""))

    save_results_with_same_name_from_folder_in_first_num_of_file_name_order(test_path, name_value_delimiter=":", save_folder_path=test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
    save_results_with_same_name_from_folder_in_first_num_of_file_name_order(val_path, name_value_delimiter=":", save_folder_path=val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
    dataset_dominant_pos_rate = -1
    dataset_sensitive_pos_rate = -1
    with open(os.path.join(data_folder, "dominant_positive_rate.txt"), 'r') as f_1, \
        open(os.path.join(data_folder, "sensitive_positive_rate.txt"), 'r') as f_2:
        dataset_dominant_pos_rate = float(f_1.readline().strip())
        dataset_sensitive_pos_rate = float(f_2.readline().strip())

    if compare_name:

        if compare_name == "Folds random removals":
            compare_test_path = os.path.join(data_folder, "test_results/10_fold_scv", experiment_folder, "random_removals_equal_to_alg_folds_10_fold_scv/")
            compare_test_save_path = os.path.join(data_folder, "test_results/10_fold_scv", experiment_folder, "random_removals_equal_to_alg_folds_10_fold_scv_over_var/")
            compare_val_path = os.path.join(data_folder, "val_results/10_fold_scv", experiment_folder, "random_removals_equal_to_alg_folds_10_fold_scv/")
            compare_val_save_path = os.path.join(data_folder, "val_results/10_fold_scv", experiment_folder, "random_removals_equal_to_alg_folds_10_fold_scv_over_var/")
        else:
            compare_name = "Knn"
            compare_test_path = os.path.join(data_folder, "test_results/knn_10_fold_scv/")
            compare_test_save_path = os.path.join(data_folder, "test_results/knn_10_fold_scv_over_var/")
            compare_val_path = os.path.join(data_folder, "val_results/knn_10_fold_scv/")
            compare_val_save_path = os.path.join(data_folder, "val_results/knn_10_fold_scv_over_var/")
        
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(compare_test_path, name_value_delimiter=":", save_folder_path=compare_test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(compare_val_path, name_value_delimiter=":", save_folder_path=compare_val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        
        if add_y_line_from_comparison_over_var_file_line == None:

            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_dataset_acc.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_dataset_acc.txt"),
                                            os.path.join(val_save_path, "avg_val_dataset_acc.txt"),
                                            os.path.join(test_save_path, "avg_test_dataset_acc.txt")],
                                            [None, None, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Accuracy",
                                            x_label,
                                            "Rate",
                                            [compare_name + " accuracy for validation set", compare_name + " accuracy for testing set",
                                             "Algorithm accuracy for validation set", "Algorithm accuracy for testing set"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_pos_prec.txt"),
                                            os.path.join(compare_val_save_path, "avg_val_pos_rec.txt"),
                                            os.path.join(val_save_path, "avg_val_pos_prec.txt"),
                                            os.path.join(val_save_path, "avg_val_pos_rec.txt")],
                                            [None, None, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Precision and recall for positive class in validation set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_neg_prec.txt"),
                                            os.path.join(compare_val_save_path, "avg_val_neg_rec.txt"),
                                            os.path.join(val_save_path, "avg_val_neg_prec.txt"),
                                            os.path.join(val_save_path, "avg_val_neg_rec.txt")],
                                            [None, None, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Precision and recall for negative class in validation set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_test_save_path, "avg_test_pos_prec.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_pos_rec.txt"),
                                            os.path.join(test_save_path, "avg_test_pos_prec.txt"),
                                            os.path.join(test_save_path, "avg_test_pos_rec.txt")],
                                            [None, None, None, None],
                                            os.path.join(test_save_path, "x_values.txt"),
                                            "Precision and recall for positive class in testing set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_test_save_path, "avg_test_neg_prec.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_neg_rec.txt"),
                                            os.path.join(test_save_path, "avg_test_neg_prec.txt"),
                                            os.path.join(test_save_path, "avg_test_neg_rec.txt")],
                                            [None, None, None, None],
                                            os.path.join(test_save_path, "x_values.txt"),
                                            "Precision and recall for negative class in testing set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_difference_of_files([(os.path.join(compare_val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(compare_val_save_path, "avg_val_sens_pos_percent.txt")),
                                        (os.path.join(compare_test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(compare_test_save_path, "avg_test_sens_pos_percent.txt")),
                                        (os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(val_save_path, "avg_val_sens_pos_percent.txt")),
                                        (os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(test_save_path, "avg_test_sens_pos_percent.txt"))],
                                        [[None, None], [None, None], [None, None], [None, None]],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Difference of dominant and sensitive positive rates",
                                        x_label,
                                        "Difference",
                                        [compare_name + " difference in validation set", compare_name + " difference in testing set",
                                        "Algorithm difference in validation set", "Algorithm difference in testing set"],
                                        [dataset_dominant_pos_rate - dataset_sensitive_pos_rate],
                                        ["Dataset difference"],
                                        plots_folder,
                                        False)
            
            plot_difference_of_files([(os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(compare_val_save_path, "avg_val_dom_pos_percent.txt")),
                                        (os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(compare_test_save_path, "avg_test_dom_pos_percent.txt")),
                                        (os.path.join(val_save_path, "avg_val_sens_pos_percent.txt"),
                                        os.path.join(compare_val_save_path, "avg_val_sens_pos_percent.txt")),
                                        (os.path.join(test_save_path, "avg_test_sens_pos_percent.txt"),
                                        os.path.join(compare_test_save_path, "avg_test_sens_pos_percent.txt"))],
                                        [[None, None], [None, None], [None, None], [None, None]],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Difference between algorithm and " + str.lower(compare_name) + " positive rates",
                                        x_label,
                                        "Difference",
                                        ["Dominant rates difference in validation set", "Dominant rates difference in testing set",
                                        "Sensitive rates difference in validation set", "Sensitive rates difference in testing set"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)

            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_dom_pos_percent.txt"),
                                            os.path.join(compare_val_save_path, "avg_val_sens_pos_percent.txt"),
                                            os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                            os.path.join(val_save_path, "avg_val_sens_pos_percent.txt")],
                                            [None, None, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Dominant vs sensitive positive rates in validation set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " dominant positive rate", compare_name + " sensitive positive rate", "Algorithm dominant positive rate", "Algorithm sensitive positive rate"],
                                            [dataset_dominant_pos_rate, dataset_sensitive_pos_rate],
                                            ["Dataset dominant positive rate", "Dataset sensitive positive rate"],
                                            plots_folder,
                                            False)

            plot_results_together_from_files([os.path.join(compare_test_save_path, "avg_test_dom_pos_percent.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_sens_pos_percent.txt"),
                                            os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                            os.path.join(test_save_path, "avg_test_sens_pos_percent.txt")],
                                            [None, None, None, None],
                                            os.path.join(test_save_path, "x_values.txt"),
                                            "Dominant vs sensitive positive rates in testing set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " dominant positive rate", compare_name + " sensitive positive rate", "Algorithm dominant positive rate", "Algorithm sensitive positive rate"],
                                            [dataset_dominant_pos_rate, dataset_sensitive_pos_rate],
                                            ["Dataset dominant positive rate", "Dataset sensitive positive rate"],
                                            plots_folder,
                                            False)
        # if it's over some variable with specific k value
        else:
            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_dataset_acc.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_dataset_acc.txt"),
                                            os.path.join(val_save_path, "avg_val_dataset_acc.txt"),
                                            os.path.join(test_save_path, "avg_test_dataset_acc.txt")],
                                            [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Accuracy",
                                            x_label,
                                            "Rate",
                                            [compare_name + " accuracy for validation set", compare_name + " accuracy for testing set",
                                             "Algorithm accuracy for validation set", "Algorithm accuracy for testing set"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_pos_prec.txt"),
                                            os.path.join(compare_val_save_path, "avg_val_pos_rec.txt"),
                                            os.path.join(val_save_path, "avg_val_pos_prec.txt"),
                                            os.path.join(val_save_path, "avg_val_pos_rec.txt")],
                                            [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Precision and recall for positive class in validation set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_neg_prec.txt"),
                                            os.path.join(compare_val_save_path, "avg_val_neg_rec.txt"),
                                            os.path.join(val_save_path, "avg_val_neg_prec.txt"),
                                            os.path.join(val_save_path, "avg_val_neg_rec.txt")],
                                            [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Precision and recall for negative class in validation set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_test_save_path, "avg_test_pos_prec.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_pos_rec.txt"),
                                            os.path.join(test_save_path, "avg_test_pos_prec.txt"),
                                            os.path.join(test_save_path, "avg_test_pos_rec.txt")],
                                            [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line, None, None],
                                            os.path.join(test_save_path, "x_values.txt"),
                                            "Precision and recall for positive class in testing set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_results_together_from_files([os.path.join(compare_test_save_path, "avg_test_neg_prec.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_neg_rec.txt"),
                                            os.path.join(test_save_path, "avg_test_neg_prec.txt"),
                                            os.path.join(test_save_path, "avg_test_neg_rec.txt")],
                                            [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line, None, None],
                                            os.path.join(test_save_path, "x_values.txt"),
                                            "Precision and recall for negative class in testing set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " precision", compare_name + " recall", "Algorithm precision", "Algorithm recall"],
                                            [],
                                            [],
                                            plots_folder,
                                            False)
            
            plot_difference_of_files([(os.path.join(compare_val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(compare_val_save_path, "avg_val_sens_pos_percent.txt")),
                                        (os.path.join(compare_test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(compare_test_save_path, "avg_test_sens_pos_percent.txt")),
                                        (os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(val_save_path, "avg_val_sens_pos_percent.txt")),
                                        (os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(test_save_path, "avg_test_sens_pos_percent.txt"))],
                                        [[add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line], [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line], 
                                         [None, None], [None, None]],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Difference of dominant and sensitive positive rates",
                                        x_label,
                                        "Difference",
                                        [compare_name + " difference in validation set", compare_name + " difference in testing set",
                                        "Algorithm difference in validation set", "Algorithm difference in testing set"],
                                        [dataset_dominant_pos_rate - dataset_sensitive_pos_rate],
                                        ["Dataset difference"],
                                        plots_folder,
                                        False)
            
            plot_difference_of_files([(os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(compare_val_save_path, "avg_val_dom_pos_percent.txt")),
                                        (os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(compare_test_save_path, "avg_test_dom_pos_percent.txt")),
                                        (os.path.join(val_save_path, "avg_val_sens_pos_percent.txt"),
                                        os.path.join(compare_val_save_path, "avg_val_sens_pos_percent.txt")),
                                        (os.path.join(test_save_path, "avg_test_sens_pos_percent.txt"),
                                        os.path.join(compare_test_save_path, "avg_test_sens_pos_percent.txt"))],
                                        [[None, add_y_line_from_comparison_over_var_file_line], [None, add_y_line_from_comparison_over_var_file_line], [None, add_y_line_from_comparison_over_var_file_line], [None, add_y_line_from_comparison_over_var_file_line]],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Difference of algorithm and " + str.lower(compare_name) +" positive rates",
                                        x_label,
                                        "Difference",
                                        ["Dominant rates difference in validation set", "Dominant rates difference in testing set",
                                        "Sensitive rates difference in validation set", "Sensitive rates difference in testing set"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)

            plot_results_together_from_files([os.path.join(compare_val_save_path, "avg_val_dom_pos_percent.txt"),
                                            os.path.join(compare_val_save_path, "avg_val_sens_pos_percent.txt"),
                                            os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                            os.path.join(val_save_path, "avg_val_sens_pos_percent.txt")],
                                            [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line, None, None],
                                            os.path.join(val_save_path, "x_values.txt"),
                                            "Dominant vs sensitive positive rates in validation set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " dominant positive rate", compare_name + " sensitive positive rate", "Algorithm dominant positive rate", "Algorithm sensitive positive rate"],
                                            [dataset_dominant_pos_rate, dataset_sensitive_pos_rate],
                                            ["Dataset dominant positive rate", "Dataset sensitive positive rate"],
                                            plots_folder,
                                            False)

            plot_results_together_from_files([os.path.join(compare_test_save_path, "avg_test_dom_pos_percent.txt"),
                                            os.path.join(compare_test_save_path, "avg_test_sens_pos_percent.txt"),
                                            os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                            os.path.join(test_save_path, "avg_test_sens_pos_percent.txt")],
                                            [add_y_line_from_comparison_over_var_file_line, add_y_line_from_comparison_over_var_file_line, None, None],
                                            os.path.join(test_save_path, "x_values.txt"),
                                            "Dominant vs sensitive positive rates in testing set",
                                            x_label,
                                            "Rate",
                                            [compare_name + " dominant positive rate", compare_name + " sensitive positive rate", "Algorithm dominant positive rate", "Algorithm sensitive positive rate"],
                                            [dataset_dominant_pos_rate, dataset_sensitive_pos_rate],
                                            ["Dataset dominant positive rate", "Dataset sensitive positive rate"],
                                            plots_folder,
                                            False)
    # if no comparison
    else:

        plot_results_together_from_files([os.path.join(val_save_path, "avg_val_dataset_acc.txt"),
                                          os.path.join(test_save_path, "avg_test_dataset_acc.txt")],
                                        [None, None],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Accuracy",
                                        x_label,
                                        "Rate",
                                        ["Validation set", "Testing set"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(val_save_path, "avg_val_pos_prec.txt"),
                                        os.path.join(val_save_path, "avg_val_pos_rec.txt")],
                                        [None, None],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Precision and recall for positive class in validation set",
                                        x_label,
                                        "Rate",
                                        ["Precision", "Recall"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(val_save_path, "avg_val_neg_prec.txt"),
                                        os.path.join(val_save_path, "avg_val_neg_rec.txt")],
                                        [None, None],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Precision and recall for negative class in validation set",
                                        x_label,
                                        "Rate",
                                        ["Precision", "Recall"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(test_save_path, "avg_test_pos_prec.txt"),
                                        os.path.join(test_save_path, "avg_test_pos_rec.txt")],
                                        [None, None],
                                        os.path.join(test_save_path, "x_values.txt"),
                                        "Precision and recall for positive class in testing set",
                                        x_label,
                                        "Rate",
                                        ["Precision", "Recall"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(test_save_path, "avg_test_neg_prec.txt"),
                                        os.path.join(test_save_path, "avg_test_neg_rec.txt")],
                                        [None, None],
                                        os.path.join(test_save_path, "x_values.txt"),
                                        "Precision and recall for negative class in testing set",
                                        x_label,
                                        "Rate",
                                        ["Precision", "Recall"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_difference_of_files([(os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                os.path.join(val_save_path, "avg_val_sens_pos_percent.txt")),
                                (os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                os.path.join(test_save_path, "avg_test_sens_pos_percent.txt"))],
                                [[None, None], [None, None]],
                                os.path.join(test_save_path, "x_values.txt"),
                                "Difference of dominant and sensitive positive rates",
                                x_label,
                                "Difference",
                                ["Algorithm difference in validation set", "Algorithm difference in testing set"],
                                [dataset_dominant_pos_rate - dataset_sensitive_pos_rate],
                                ["Dataset difference"],
                                plots_folder,
                                False)

        plot_results_together_from_files([os.path.join(val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(val_save_path, "avg_val_sens_pos_percent.txt")],
                                        [None, None],
                                        os.path.join(val_save_path, "x_values.txt"),
                                        "Dominant vs sensitive positive rates in validation set",
                                        x_label,
                                        "Rate",
                                        ["Dominant percent", "Sensitive percent"],
                                        [dataset_dominant_pos_rate, dataset_sensitive_pos_rate],
                                        ["Dataset dominant positive rate", "Dataset sensitive positive rate"],
                                        plots_folder,
                                        False)

        plot_results_together_from_files([os.path.join(test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(test_save_path, "avg_test_sens_pos_percent.txt")],
                                        [None, None],
                                        os.path.join(test_save_path, "x_values.txt"),
                                        "Dominant vs sensitive positive rates in testing set",
                                        x_label,
                                        "Rate",
                                        ["Dominant percent", "Sensitive percent"],
                                        [dataset_dominant_pos_rate, dataset_sensitive_pos_rate],
                                        ["Dataset dominant positive rate", "Dataset sensitive positive rate"],
                                        plots_folder,
                                        False)
        
    plot_results_together_from_files([os.path.join(val_save_path, "avg_train_ids_removed.txt")],
                                    [None],
                                    os.path.join(val_save_path, "x_values.txt"),
                                    "Average train points removed",
                                    x_label,
                                    "Value",
                                    None,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(val_save_path, "avg_percentage_of_train_ids_removed.txt")],
                                    [None],
                                    os.path.join(val_save_path, "x_values.txt"),
                                    "Average rate of train points removed",
                                    x_label,
                                    "Rate",
                                    None,
                                    [],
                                    [],
                                    plots_folder,
                                    False)

    plot_results_together_from_files([os.path.join(val_save_path, "avg_val_first_pred_acc.txt"),
                                    os.path.join(test_save_path, "avg_test_first_pred_acc.txt")],
                                    [None, None],
                                    os.path.join(val_save_path, "x_values.txt"),
                                    "Algorithm prediction similarity to knn",
                                    x_label,
                                    "Rate",
                                    ["Validation set", "Testing set"],
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
def save_knn_plots_of_multiple_datasets(data_folders, legend_labels, plots_folder, save_every, num_smaller_than, only_div_by=None, num_greater_than=0):
    knn_test_paths = []
    knn_test_save_paths = []
    knn_val_paths = []
    knn_val_save_paths = []
    for i, data_folder in enumerate(data_folders):
        knn_test_paths.append(os.path.join(data_folder, "test_results/knn_10_fold_scv/"))
        knn_test_save_paths.append(os.path.join(data_folder, "test_results/knn_10_fold_scv_over_var/"))
        knn_val_paths.append(os.path.join(data_folder, "val_results/knn_10_fold_scv/"))
        knn_val_save_paths.append(os.path.join(data_folder, "val_results/knn_10_fold_scv_over_var/"))

        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(knn_test_paths[i], name_value_delimiter=":", save_folder_path=knn_test_save_paths[i], save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(knn_val_paths[i], name_value_delimiter=":", save_folder_path=knn_val_save_paths[i], save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)

    plot_results_together_from_files([os.path.join(knn_val_save_paths[i], "avg_val_dataset_acc.txt") for i in range(len(knn_val_save_paths))],
                                    [None] * len(knn_val_save_paths),
                                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                                    "Knn Accuracy",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_test_save_paths[i], "avg_test_dataset_acc.txt") for i in range(len(knn_test_save_paths))],
                                    [None] * len(knn_test_save_paths),
                                    os.path.join(knn_test_save_paths[0], "x_values.txt"),
                                    "Knn Accuracy for Testing Set",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)

    plot_results_together_from_files([os.path.join(knn_val_save_paths[i], "avg_val_pos_prec.txt") for i in range(len(knn_val_save_paths))],
                                    [None] * len(knn_val_save_paths),
                                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                                    "Knn Precision for Positive Class",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_test_save_paths[i], "avg_test_pos_prec.txt") for i in range(len(knn_test_save_paths))],
                                    [None] * len(knn_test_save_paths),
                                    os.path.join(knn_test_save_paths[0], "x_values.txt"),
                                    "Knn Precision for Positive Class in Testing Set",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_val_save_paths[i], "avg_val_neg_prec.txt") for i in range(len(knn_val_save_paths))],
                                    [None] * len(knn_val_save_paths),
                                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                                    "Knn Precision for Negative Class",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_test_save_paths[i], "avg_test_neg_prec.txt") for i in range(len(knn_test_save_paths))],
                                    [None] * len(knn_test_save_paths),
                                    os.path.join(knn_test_save_paths[0], "x_values.txt"),
                                    "Knn Precision for Negative Class in Testing Set",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_val_save_paths[i], "avg_val_pos_rec.txt") for i in range(len(knn_val_save_paths))],
                                    [None] * len(knn_val_save_paths),
                                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                                    "Knn Recall for Positive Class",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_test_save_paths[i], "avg_test_pos_rec.txt") for i in range(len(knn_test_save_paths))],
                                    [None] * len(knn_test_save_paths),
                                    os.path.join(knn_test_save_paths[0], "x_values.txt"),
                                    "Knn Recall for Positive Class in Testing Set",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_val_save_paths[i], "avg_val_neg_rec.txt") for i in range(len(knn_val_save_paths))],
                                    [None] * len(knn_val_save_paths),
                                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                                    "Knn Recall for Negative Class",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_test_save_paths[i], "avg_test_neg_rec.txt") for i in range(len(knn_test_save_paths))],
                                    [None] * len(knn_test_save_paths),
                                    os.path.join(knn_test_save_paths[0], "x_values.txt"),
                                    "Knn Recall for Negative Class in Testing Set",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)

    plot_results_together_from_files([os.path.join(knn_val_save_paths[i], "avg_val_dom_pos_percent.txt") for i in range(len(knn_val_save_paths))],
                                    [None] * len(knn_val_save_paths),
                                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                                    "Knn Dominant Positive Rates",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_test_save_paths[i], "avg_test_dom_pos_percent.txt") for i in range(len(knn_test_save_paths))],
                                    [None] * len(knn_test_save_paths),
                                    os.path.join(knn_test_save_paths[0], "x_values.txt"),
                                    "Knn Dominant Positive Rates in Test set",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_val_save_paths[i], "avg_val_sens_pos_percent.txt") for i in range(len(knn_val_save_paths))],
                                    [None] * len(knn_val_save_paths),
                                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                                    "Knn Sensitive Positive Rates",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_results_together_from_files([os.path.join(knn_test_save_paths[i], "avg_test_sens_pos_percent.txt") for i in range(len(knn_test_save_paths))],
                                    [None] * len(knn_test_save_paths),
                                    os.path.join(knn_test_save_paths[0], "x_values.txt"),
                                    "Knn Sensitive Positive Rates in Test set",
                                    "K Value",
                                    "Rate",
                                    legend_labels,
                                    [],
                                    [],
                                    plots_folder,
                                    False)
    
    plot_difference_of_files([(os.path.join(knn_val_save_path, "avg_val_dom_pos_percent.txt"),
                            os.path.join(knn_val_save_path, "avg_val_sens_pos_percent.txt")) for knn_val_save_path in knn_val_save_paths],
                            [[None, None]] * len(knn_val_save_paths),
                            os.path.join(knn_val_save_paths[0], "x_values.txt"),
                            "Knn Difference Between Dominant and Sensitive Positive Rates",
                            "K Value",
                            "Difference",
                            legend_labels,
                            [],
                            [],
                            plots_folder,
                            False)
    
    plot_difference_of_files([(os.path.join(knn_test_save_path, "avg_test_dom_pos_percent.txt"),
                            os.path.join(knn_test_save_path, "avg_test_sens_pos_percent.txt")) for knn_test_save_path in knn_test_save_paths],
                            [[None, None]] * len(knn_test_save_paths),
                            os.path.join(knn_val_save_paths[0], "x_values.txt"),
                            "Knn Difference Between Dominant and Sensitive Positive Rates in Testing Set",
                            "K Value",
                            "Difference",
                            legend_labels,
                            [],
                            [],
                            plots_folder,
                            False)
    
    plot_f1_of_files([os.path.join(knn_val_save_path, "avg_val_pos_prec.txt") for knn_val_save_path in knn_val_save_paths],
                    [os.path.join(knn_val_save_path, "avg_val_pos_rec.txt") for knn_val_save_path in knn_val_save_paths],
                    None,
                    None,
                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                    "Knn F1 Score for the Positive Class",
                    "K Value",
                    "Score",
                    legend_labels,
                    [],
                    [],
                    plots_folder,
                    False)
    
    plot_f1_of_files([os.path.join(knn_val_save_path, "avg_val_neg_prec.txt") for knn_val_save_path in knn_val_save_paths],
                    [os.path.join(knn_val_save_path, "avg_val_neg_rec.txt") for knn_val_save_path in knn_val_save_paths],
                    None,
                    None,
                    os.path.join(knn_val_save_paths[0], "x_values.txt"),
                    "Knn F1 Score for the Negative Class",
                    "K Value",
                    "Score",
                    legend_labels,
                    [],
                    [],
                    plots_folder,
                    False)

def save_alg_1_bar_plots(data_folders, k_value, file_line_for_bar_plots, experiment_dif, save_every, num_smaller_than, only_div_by=None, num_greater_than=0):
    for data_folder in data_folders:
        knn_test_path = os.path.join(data_folder, "test_results/knn_10_fold_scv/")
        knn_test_save_path = os.path.join(data_folder, "test_results/knn_10_fold_scv_over_var/")
        knn_val_path = os.path.join(data_folder, "val_results/knn_10_fold_scv/")
        knn_val_save_path = os.path.join(data_folder, "val_results/knn_10_fold_scv_over_var/")
        alg_test_path = os.path.join(data_folder, "test_results/10_fold_scv/over_k/perc_dif_" + str(experiment_dif) + "/")
        alg_test_save_path = os.path.join(data_folder, "test_results/10_fold_scv_over_var/over_k/perc_dif_" + str(experiment_dif) + "/")
        alg_val_path = os.path.join(data_folder, "val_results/10_fold_scv/over_k/perc_dif_" + str(experiment_dif) + "/")
        alg_val_save_path = os.path.join(data_folder, "val_results/10_fold_scv_over_var/over_k/perc_dif_" + str(experiment_dif) + "/")
        rand_until_fair_test_path = os.path.join(data_folder, "test_results/10_fold_scv/over_k/perc_dif_" + str(experiment_dif) + "/random_removals_until_fair_10_fold_scv/")
        rand_until_fair_test_save_path = os.path.join(data_folder, "test_results/10_fold_scv_over_var/over_k/perc_dif_" + str(experiment_dif) + "/random_removals_until_fair_10_fold_scv/")
        rand_until_fair_val_path = os.path.join(data_folder, "val_results/10_fold_scv/over_k/perc_dif_" + str(experiment_dif) + "/random_removals_until_fair_10_fold_scv/")
        rand_until_fair_val_save_path = os.path.join(data_folder, "val_results/10_fold_scv_over_var/over_k/perc_dif_" + str(experiment_dif) + "/random_removals_until_fair_10_fold_scv/")
        barplots_folder = os.path.join(data_folder, "plots/" + str(k_value) + "_neigh", "perc_dif_" + str(experiment_dif) + "/")
        
        dataset_dominant_pos_rate = -1
        dataset_sensitive_pos_rate = -1
        with open(os.path.join(data_folder, "dominant_positive_rate.txt"), 'r') as f_1, \
            open(os.path.join(data_folder, "sensitive_positive_rate.txt"), 'r') as f_2:
            dataset_dominant_pos_rate = float(f_1.readline().strip())
            dataset_sensitive_pos_rate = float(f_2.readline().strip())

        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(knn_test_path, name_value_delimiter=":", save_folder_path=knn_test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(knn_val_path, name_value_delimiter=":", save_folder_path=knn_val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(alg_test_path, name_value_delimiter=":", save_folder_path=alg_test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(alg_val_path, name_value_delimiter=":", save_folder_path=alg_val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(rand_until_fair_test_path, name_value_delimiter=":", save_folder_path=rand_until_fair_test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(rand_until_fair_val_path, name_value_delimiter=":", save_folder_path=rand_until_fair_val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)

        with open(os.path.join(knn_val_save_path, "avg_val_dom_pos_percent.txt"), 'r') as knn_val_dom_f, \
             open(os.path.join(alg_val_save_path, "avg_val_dom_pos_percent.txt"), 'r') as alg_val_dom_f, \
             open(os.path.join(rand_until_fair_val_save_path, "avg_val_dom_pos_percent.txt"), 'r') as rand_until_fair_val_dom_f, \
             open(os.path.join(knn_val_save_path, "avg_val_sens_pos_percent.txt"), 'r') as knn_val_sens_f, \
             open(os.path.join(alg_val_save_path, "avg_val_sens_pos_percent.txt"), 'r') as alg_val_sens_f, \
             open(os.path.join(rand_until_fair_val_save_path, "avg_val_sens_pos_percent.txt"), 'r') as rand_until_fair_val_sens_f, \
             open(os.path.join(knn_test_save_path, "avg_test_dom_pos_percent.txt"), 'r') as knn_test_dom_f, \
             open(os.path.join(alg_test_save_path, "avg_test_dom_pos_percent.txt"), 'r') as alg_test_dom_f, \
             open(os.path.join(rand_until_fair_test_save_path, "avg_test_dom_pos_percent.txt"), 'r') as rand_until_fair_test_dom_f, \
             open(os.path.join(knn_test_save_path, "avg_test_sens_pos_percent.txt"), 'r') as knn_test_sens_f, \
             open(os.path.join(alg_test_save_path, "avg_test_sens_pos_percent.txt"), 'r') as alg_test_sens_f, \
             open(os.path.join(rand_until_fair_test_save_path, "avg_test_sens_pos_percent.txt"), 'r') as rand_until_fair_test_sens_f, \
             open(os.path.join(alg_val_save_path, "avg_train_ids_removed.txt"), 'r') as alg_ids_removed_f, \
             open(os.path.join(rand_until_fair_val_save_path, "avg_train_ids_removed.txt"), 'r') as rand_until_fair_ids_removed_f, \
             open(os.path.join(alg_val_save_path, "avg_percentage_of_train_ids_removed.txt"), 'r') as alg_rate_ids_removed_f, \
             open(os.path.join(rand_until_fair_val_save_path, "avg_percentage_of_train_ids_removed.txt"), 'r') as rand_until_fair_rate_ids_removed_f:
            
            knn_val_dom_barplot_results = [float(line.strip()) for (j, line) in enumerate(knn_val_dom_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            alg_val_dom_barplot_results = [float(line.strip()) for (j, line) in enumerate(alg_val_dom_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            rand_until_fair_val_dom_barplot_results = [float(line.strip()) for (j, line) in enumerate(rand_until_fair_val_dom_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            knn_val_sens_barplot_results = [float(line.strip()) for (j, line) in enumerate(knn_val_sens_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            alg_val_sens_barplot_results = [float(line.strip()) for (j, line) in enumerate(alg_val_sens_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            rand_until_fair_val_sens_barplot_results = [float(line.strip()) for (j, line) in enumerate(rand_until_fair_val_sens_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            knn_test_dom_barplot_results = [float(line.strip()) for (j, line) in enumerate(knn_test_dom_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            alg_test_dom_barplot_results = [float(line.strip()) for (j, line) in enumerate(alg_test_dom_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            rand_until_fair_test_dom_barplot_results = [float(line.strip()) for (j, line) in enumerate(rand_until_fair_test_dom_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            knn_test_sens_barplot_results = [float(line.strip()) for (j, line) in enumerate(knn_test_sens_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            alg_test_sens_barplot_results = [float(line.strip()) for (j, line) in enumerate(alg_test_sens_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            rand_until_fair_test_sens_barplot_results = [float(line.strip()) for (j, line) in enumerate(rand_until_fair_test_sens_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            alg_ids_removed_barplot_results = [float(line.strip()) for (j, line) in enumerate(alg_ids_removed_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            rand_until_fair_ids_removed_barplot_results = [float(line.strip()) for (j, line) in enumerate(rand_until_fair_ids_removed_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            alg_rate_ids_removed_barplot_results = [float(line.strip()) for (j, line) in enumerate(alg_rate_ids_removed_f) if (line.strip() and j==file_line_for_bar_plots)][0]
            rand_until_fair_rate_ids_removed_barplot_results = [float(line.strip()) for (j, line) in enumerate(rand_until_fair_rate_ids_removed_f) if (line.strip() and j==file_line_for_bar_plots)][0]

            knn_val_dom_sens_dif_for_barplot = knn_val_dom_barplot_results - knn_val_sens_barplot_results
            alg_val_dom_sens_dif_for_barplot = alg_val_dom_barplot_results - alg_val_sens_barplot_results
            rand_until_fair_val_dom_sens_dif_for_barplot = rand_until_fair_val_dom_barplot_results - rand_until_fair_val_sens_barplot_results
            knn_test_dom_sens_dif_for_barplot = knn_test_dom_barplot_results - knn_test_sens_barplot_results
            alg_test_dom_sens_dif_for_barplot = alg_test_dom_barplot_results - alg_test_sens_barplot_results
            rand_until_fair_test_dom_sens_dif_for_barplot = rand_until_fair_test_dom_barplot_results - rand_until_fair_test_sens_barplot_results
            
            plt.figure(figsize=(6.4, 4.8))
            bars = plt.bar(["Dataset", "Knn\n(Validation Set)", "Min-Rem-Parity\n(Validation Set)", "Random Removal\n(Validation Set)"],
                            [dataset_dominant_pos_rate, knn_val_dom_barplot_results, alg_val_dom_barplot_results, rand_until_fair_val_dom_barplot_results],
                            color = plt.rcParams['axes.prop_cycle'].by_key()['color'])
            plt.ylabel("Rate")
            plt.title("Dominant Positive Rate")
            plt.gca().bar_label(bars, fmt='%.2f')
            if not os.path.exists(barplots_folder):
                os.makedirs(barplots_folder)
            plt.savefig(os.path.join(barplots_folder, "Dominant Positive Rate"), bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6.4, 4.8))
            bars = plt.bar(["Dataset", "Knn\n(Testing Set)", "Min-Rem-Parity\n(Testing Set)", "Random Removal\n(Testing Set)"],
                            [dataset_dominant_pos_rate, knn_test_dom_barplot_results, alg_test_dom_barplot_results, rand_until_fair_test_dom_barplot_results],
                            color = plt.rcParams['axes.prop_cycle'].by_key()['color'])
            plt.ylabel("Rate")
            plt.title("Dominant Positive Rate")
            plt.gca().bar_label(bars, fmt='%.2f')
            plt.savefig(os.path.join(barplots_folder, "Dominant Positive Rate in Testing"), bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(3.2, 4.8))
            bars = plt.bar([0, 0.33],
                    [dataset_dominant_pos_rate - dataset_sensitive_pos_rate, knn_val_dom_sens_dif_for_barplot],
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'],
                    width=0.25)
            plt.xticks([0, 0.33], ["Dataset", "Knn\n(Validation Set)"])
            plt.ylabel("Difference")
            plt.title("Difference Between Dominant and \nSensitive Positive Rates\n")
            plt.gca().bar_label(bars, fmt='%.2f')
            plt.savefig(os.path.join(barplots_folder, "Difference Between Dominant and Sensitive Positive Rates in Validation Set"), bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6.4, 4.8))
            bars = plt.bar(["Dataset", "Knn\n(Testing Set)", "Min-Rem-Parity\n(Testing Set)", "Random Removal\n(Testing Set)"],
                    [dataset_dominant_pos_rate - dataset_sensitive_pos_rate, knn_test_dom_sens_dif_for_barplot, alg_test_dom_sens_dif_for_barplot, rand_until_fair_test_dom_sens_dif_for_barplot],
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'])
            plt.ylabel("Difference")
            plt.title("Difference Between Dominant and Sensitive Positive Rates\n")
            plt.gca().bar_label(bars, fmt='%.2f')
            plt.savefig(os.path.join(barplots_folder, "Difference Between Dominant and Sensitive Positive Rates in Testing Set"), bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(3.2, 4.8))
            bars = plt.bar([0, 0.33],
                    [alg_ids_removed_barplot_results, rand_until_fair_ids_removed_barplot_results],
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'],
                    width=0.25)
            plt.xticks([0, 0.33], ["Min-Rem-Parity", "Random Removal"])
            plt.ylabel("Value")
            plt.title("Training Points Removed\n")
            plt.gca().bar_label(bars, fmt='%.2f')
            plt.savefig(os.path.join(barplots_folder, "Training Points Removed"), bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(3.2, 4.8))
            bars = plt.bar([0, 0.33],
                    [alg_rate_ids_removed_barplot_results, rand_until_fair_rate_ids_removed_barplot_results],
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'],
                    width=0.25)
            plt.xticks([0, 0.33], ["Min-Rem-Parity", "Random Removal"])
            plt.ylabel("Rate")
            plt.title("Training Point Removal Rate\n")
            plt.gca().bar_label(bars, fmt='%.2f')
            plt.savefig(os.path.join(barplots_folder, "Training Point Removal Rate"), bbox_inches='tight')
            plt.close()

def save_plots_over_var(data_folders, experiment_path, x_label, add_y_line_from_knn_over_k_file_line, save_every, num_smaller_than, only_div_by=None, num_greater_than=0):

    for data_folder in data_folders:
        knn_test_path = os.path.join(data_folder, "test_results/knn_10_fold_scv/")
        knn_test_save_path =  os.path.join(data_folder, "test_results/knn_10_fold_scv_over_var/")
        knn_val_path = os.path.join(data_folder, "val_results/knn_10_fold_scv/")
        knn_val_save_path = os.path.join(data_folder, "val_results/knn_10_fold_scv_over_var/")
        alg_test_path = os.path.join(data_folder, "test_results/10_fold_scv/" + experiment_path)
        alg_test_save_path = os.path.join(data_folder, "test_results/10_fold_scv_over_var/" + experiment_path)
        alg_val_path = os.path.join(data_folder, "val_results/10_fold_scv/" + experiment_path)
        alg_val_save_path = os.path.join(data_folder, "val_results/10_fold_scv_over_var/" + experiment_path)
        rand_until_fair_test_path = os.path.join(data_folder, "test_results/10_fold_scv/" + experiment_path + "random_removals_until_fair_10_fold_scv/")
        rand_until_fair_test_save_path = os.path.join(data_folder, "test_results/10_fold_scv_over_var/" + experiment_path + "random_removals_until_fair_10_fold_scv/")
        rand_until_fair_val_path = os.path.join(data_folder, "val_results/10_fold_scv/" + experiment_path+ "random_removals_until_fair_10_fold_scv/")
        rand_until_fair_val_save_path = os.path.join(data_folder, "val_results/10_fold_scv_over_var/" + experiment_path + "random_removals_until_fair_10_fold_scv/")
        plots_folder = os.path.join(data_folder, "plots/" + experiment_path)
            
        dataset_dominant_pos_rate = -1
        dataset_sensitive_pos_rate = -1
        with open(os.path.join(data_folder, "dominant_positive_rate.txt"), 'r') as f_1, \
            open(os.path.join(data_folder, "sensitive_positive_rate.txt"), 'r') as f_2:
            dataset_dominant_pos_rate = float(f_1.readline().strip())
            dataset_sensitive_pos_rate = float(f_2.readline().strip())

        knn_line_to_add = None
        if add_y_line_from_knn_over_k_file_line:
            knn_line_to_add = add_y_line_from_knn_over_k_file_line

        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(knn_test_path, name_value_delimiter=":", save_folder_path=knn_test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(knn_val_path, name_value_delimiter=":", save_folder_path=knn_val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(alg_test_path, name_value_delimiter=":", save_folder_path=alg_test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(alg_val_path, name_value_delimiter=":", save_folder_path=alg_val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(rand_until_fair_test_path, name_value_delimiter=":", save_folder_path=rand_until_fair_test_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)
        save_results_with_same_name_from_folder_in_first_num_of_file_name_order(rand_until_fair_val_path, name_value_delimiter=":", save_folder_path=rand_until_fair_val_save_path, save_every=save_every, num_smaller_than=num_smaller_than, only_div_by=only_div_by, num_greater_than=num_greater_than)

        plot_results_together_from_files([os.path.join(knn_val_save_path, "avg_val_dataset_acc.txt"),
                                        os.path.join(alg_val_save_path, "avg_val_dataset_acc.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_val_dataset_acc.txt"),
                                        os.path.join(knn_test_save_path, "avg_test_dataset_acc.txt"),
                                        os.path.join(alg_test_save_path, "avg_test_dataset_acc.txt"),
                                        os.path.join(rand_until_fair_test_save_path, "avg_test_dataset_acc.txt")],
                                        [knn_line_to_add, None, None, knn_line_to_add, None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Accuracy",
                                        x_label,
                                        "Rate",
                                        ["Knn(Validation)", "Min-Rem-Increase(Validation)", "Random Removal(Validation)",
                                         "Knn(Testing)", "Min-Rem-Increase(Testing)", "Random Removal(Testing)"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(knn_val_save_path, "avg_val_neg_prec.txt"),
                                        os.path.join(alg_val_save_path, "avg_val_neg_prec.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_val_neg_prec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Precision for Negative Class for Validation Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(knn_test_save_path, "avg_test_neg_prec.txt"),
                                        os.path.join(alg_test_save_path, "avg_test_neg_prec.txt"),
                                        os.path.join(rand_until_fair_test_save_path, "avg_test_neg_prec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_test_save_path, "x_values.txt"),
                                        "Precision for Negative Class for Testing set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)

        plot_results_together_from_files([os.path.join(knn_val_save_path, "avg_val_pos_prec.txt"),
                                        os.path.join(alg_val_save_path, "avg_val_pos_prec.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_val_pos_prec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Precision for Positive Class for Validation Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(knn_test_save_path, "avg_test_pos_prec.txt"),
                                        os.path.join(alg_test_save_path, "avg_test_pos_prec.txt"),
                                        os.path.join(rand_until_fair_test_save_path, "avg_test_pos_prec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_test_save_path, "x_values.txt"),
                                        "Precision for Positive Class for Testing Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(knn_val_save_path, "avg_val_neg_rec.txt"),
                                        os.path.join(alg_val_save_path, "avg_val_neg_rec.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_val_neg_rec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Recall for Negative Class for Validation Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(knn_test_save_path, "avg_test_neg_rec.txt"),
                                        os.path.join(alg_test_save_path, "avg_test_neg_rec.txt"),
                                        os.path.join(rand_until_fair_test_save_path, "avg_test_neg_rec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_test_save_path, "x_values.txt"),
                                        "Recall for Negative Class for Testing set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)

        plot_results_together_from_files([os.path.join(knn_val_save_path, "avg_val_pos_rec.txt"),
                                        os.path.join(alg_val_save_path, "avg_val_pos_rec.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_val_pos_rec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Recall for Positive Class for Validation Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(knn_test_save_path, "avg_test_pos_rec.txt"),
                                        os.path.join(alg_test_save_path, "avg_test_pos_rec.txt"),
                                        os.path.join(rand_until_fair_test_save_path, "avg_test_pos_rec.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_test_save_path, "x_values.txt"),
                                        "Recall for Positive Class for Testing Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_difference_of_files([(os.path.join(knn_val_save_path, "avg_val_dom_pos_percent.txt"),
                                    os.path.join(knn_val_save_path, "avg_val_sens_pos_percent.txt")),
                                    (os.path.join(alg_val_save_path, "avg_val_dom_pos_percent.txt"),
                                    os.path.join(alg_val_save_path, "avg_val_sens_pos_percent.txt")),
                                    (os.path.join(rand_until_fair_val_save_path, "avg_val_dom_pos_percent.txt"),
                                    os.path.join(rand_until_fair_val_save_path, "avg_val_sens_pos_percent.txt")),
                                    (os.path.join(knn_test_save_path, "avg_test_dom_pos_percent.txt"),
                                    os.path.join(knn_test_save_path, "avg_test_sens_pos_percent.txt")),
                                    (os.path.join(alg_test_save_path, "avg_test_dom_pos_percent.txt"),
                                    os.path.join(alg_test_save_path, "avg_test_sens_pos_percent.txt")),
                                    (os.path.join(rand_until_fair_test_save_path, "avg_test_dom_pos_percent.txt"),
                                    os.path.join(rand_until_fair_test_save_path, "avg_test_sens_pos_percent.txt"))],
                                    [[knn_line_to_add, knn_line_to_add], 
                                    [None, None], [None, None],
                                    [knn_line_to_add, knn_line_to_add], 
                                    [None, None], [None, None]],
                                    os.path.join(alg_val_save_path, "x_values.txt"),
                                    "Difference of Dominant and Sensitive Positive Rates",
                                    x_label,
                                    "Difference",
                                    ["Knn(Validation)", "Min-Rem-Increase(Validation)", "Random Removal(Validation)",
                                     "Knn(Testing)", "Min-Rem-Increase(Testing)", "Random Removal(Testing)"],
                                    [dataset_dominant_pos_rate - dataset_sensitive_pos_rate],
                                    ["Dataset Difference"],
                                    plots_folder,
                                    False)

        plot_results_together_from_files([os.path.join(knn_val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(alg_val_save_path, "avg_val_dom_pos_percent.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_val_dom_pos_percent.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Dominant Positive Rate for Validation Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [dataset_dominant_pos_rate],
                                        ["Dataset Rate"],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(knn_test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(alg_test_save_path, "avg_test_dom_pos_percent.txt"),
                                        os.path.join(rand_until_fair_test_save_path, "avg_test_dom_pos_percent.txt")],
                                        [knn_line_to_add, None, None],
                                        os.path.join(alg_test_save_path, "x_values.txt"),
                                        "Dominant Positive Rate for Testing Set",
                                        x_label,
                                        "Rate",
                                        ["Knn", "Min-Rem-Increase", "Random Removal"],
                                        [dataset_dominant_pos_rate],
                                        ["Dataset Rate"],
                                        plots_folder,
                                        False)

        plot_results_together_from_files([os.path.join(knn_val_save_path, "avg_val_sens_pos_percent.txt"),
                                        os.path.join(alg_val_save_path, "avg_val_sens_pos_percent.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_val_sens_pos_percent.txt"),
                                        os.path.join(knn_test_save_path, "avg_test_sens_pos_percent.txt"),
                                        os.path.join(alg_test_save_path, "avg_test_sens_pos_percent.txt"),
                                        os.path.join(rand_until_fair_test_save_path, "avg_test_sens_pos_percent.txt")],
                                        [knn_line_to_add, None, None, knn_line_to_add, None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Sensitive Positive Rate",
                                        x_label,
                                        "Rate",
                                        ["Knn(Validation Set)", "Min-Rem-Increase(Validation Set)", "Random Removal(Validation Set)",
                                        "Knn(Testing Set)", "Min-Rem-Increase(Testing Set)", "Random Removal(Testing Set)"],
                                        [dataset_sensitive_pos_rate],
                                        ["Dataset Rate"],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(alg_val_save_path, "avg_train_ids_removed.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_train_ids_removed.txt")],
                                        [None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Training Points Removed",
                                        x_label,
                                        "Value",
                                        ["Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_results_together_from_files([os.path.join(alg_val_save_path, "avg_percentage_of_train_ids_removed.txt"),
                                        os.path.join(rand_until_fair_val_save_path, "avg_percentage_of_train_ids_removed.txt")],
                                        [None, None],
                                        os.path.join(alg_val_save_path, "x_values.txt"),
                                        "Training Point Removal Rate",
                                        x_label,
                                        "Rate",
                                        ["Min-Rem-Increase", "Random Removal"],
                                        [],
                                        [],
                                        plots_folder,
                                        False)
        
        plot_f1_of_files([os.path.join(knn_val_save_path, "avg_val_pos_prec.txt"),
                        os.path.join(alg_val_save_path, "avg_val_pos_prec.txt"),
                        os.path.join(rand_until_fair_val_save_path, "avg_val_pos_prec.txt"),
                        os.path.join(knn_test_save_path, "avg_test_pos_prec.txt"),
                        os.path.join(alg_test_save_path, "avg_test_pos_prec.txt"),
                        os.path.join(rand_until_fair_test_save_path, "avg_test_pos_prec.txt")],
                        [os.path.join(knn_val_save_path, "avg_val_pos_rec.txt"),
                        os.path.join(alg_val_save_path, "avg_val_pos_rec.txt"),
                        os.path.join(rand_until_fair_val_save_path, "avg_val_pos_rec.txt"),
                        os.path.join(knn_test_save_path, "avg_test_pos_rec.txt"),
                        os.path.join(alg_test_save_path, "avg_test_pos_rec.txt"),
                        os.path.join(rand_until_fair_test_save_path, "avg_test_pos_rec.txt")],
                        [knn_line_to_add, None, None, knn_line_to_add, None, None],
                        [knn_line_to_add, None, None, knn_line_to_add, None, None],
                        os.path.join(alg_val_save_path, "x_values.txt"),
                        "F1 Scores for the Positive Class",
                        x_label,
                        "Score",
                        ["Knn(Validation)", "Min-Rem-Increase(Validation)", "Random Removal(Validation)",
                         "Knn(Testing)", "Min-Rem-Increase(Testing)", "Random Removal(Testing)"],
                        [],
                        [],
                        plots_folder,
                        False)
        
        plot_f1_of_files([os.path.join(knn_val_save_path, "avg_val_neg_prec.txt"),
                        os.path.join(alg_val_save_path, "avg_val_neg_prec.txt"),
                        os.path.join(rand_until_fair_val_save_path, "avg_val_neg_prec.txt"),
                        os.path.join(knn_test_save_path, "avg_test_neg_prec.txt"),
                        os.path.join(alg_test_save_path, "avg_test_neg_prec.txt"),
                        os.path.join(rand_until_fair_test_save_path, "avg_test_neg_prec.txt")],
                        [os.path.join(knn_val_save_path, "avg_val_neg_rec.txt"),
                        os.path.join(alg_val_save_path, "avg_val_neg_rec.txt"),
                        os.path.join(rand_until_fair_val_save_path, "avg_val_neg_rec.txt"),
                        os.path.join(knn_test_save_path, "avg_test_neg_rec.txt"),
                        os.path.join(alg_test_save_path, "avg_test_neg_rec.txt"),
                        os.path.join(rand_until_fair_test_save_path, "avg_test_neg_rec.txt")],
                        [knn_line_to_add, None, None, knn_line_to_add, None, None],
                        [knn_line_to_add, None, None, knn_line_to_add, None, None],
                        os.path.join(alg_val_save_path, "x_values.txt"),
                        "F1 Scores for the Negative Class",
                        x_label,
                        "Score",
                        ["Knn(Validation)", "Min-Rem-Increase(Validation)", "Random Removal(Validation)",
                         "Knn(Testing)", "Min-Rem-Increase(Testing)", "Random Removal(Testing)"],
                        [],
                        [],
                        plots_folder,
                        False)



if __name__ == "__main__":
    
    # save_plots_for_dataset("results/acs_income_gender/",
    #            "over_k/sens_incr_0.25",
    #            "K value",
    #            save_every=1,
    #            num_smaller_than=101,
    #            compare_name="Knn",  # "Folds random removals"
    #            add_y_line_from_comparison_over_var_file_line=None)
    
    # save_plots_for_dataset("results/acs_income_gender/",
    #             "over_sens_incr/21_neigh",
    #             "Sensitive positive rate increase",
    #             save_every=1,
    #             num_smaller_than=101,
    #             compare_name="Knn",
    #             add_y_line_from_comparison_over_var_file_line=10) # 10 corresponds to k=21
    
    # save_plots_for_dataset("results/acs_income_gender/",
    #            "over_k/perc_dif_0.001",
    #            "K value",
    #            save_every=1,
    #            num_smaller_than=101,
    #            compare_name="Knn",
    #            add_y_line_from_comparison_over_var_file_line=None)

    # save_knn_plots_of_multiple_datasets(["results/acs_income_gender/", "results/acs_income_race/", "results/adult_gender", "results/adult_race", "results/credit_card_default"],
    #                                     ["ACSIncome(Gender)", "ACSIncome(Race)", "Adult(Gender)", "Adult(Race)", "CreditCard(Education)"],
    #                                     "results/plots/knn",
    #                                     save_every=1,
    #                                     num_smaller_than=101)

    save_alg_1_bar_plots(["results/acs_income_gender/", "results/acs_income_race/", "results/adult_gender", "results/adult_race", "results/credit_card_default"],
                            21,
                            10,
                            0.001,
                            save_every=1,
                            num_smaller_than=101)
    
    # save_plots_over_var(["results/acs_income_gender/", "results/acs_income_race/", "results/adult_gender", "results/adult_race", "results/credit_card_default"],
    #                     "over_k/sens_incr_0.25/",
    #                     "K value",
    #                     None,
    #                     save_every=1,
    #                     num_smaller_than=101)
    
    # save_plots_over_var(["results/acs_income_gender/", "results/acs_income_race/", "results/adult_gender", "results/adult_race", "results/credit_card_default"],
    #                     "over_sens_incr/21_neigh/",
    #                     "Given Sensitive Rate Increase",
    #                     10,
    #                     save_every=1,
    #                     num_smaller_than=101)