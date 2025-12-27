import ast
import math
import random
import pandas as pd
from scipy import datasets
from sortedcontainers import SortedList
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from sklearn.neighbors import KDTree
from fairlearn import datasets
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.utilities import encode_dataframe



class ValueSortedDict:
    def __init__(self):
        self.dict = {}  # train_id -> value
        self.sorted_by_value_tuples = SortedList()  # (value, train_id) tuples sorted by value

    def add(self, key, value):
        self.dict[key] = value
        self.sorted_by_value_tuples.add((value, key))

    def update(self, key, new_value):
        old_value = self.dict[key]
        self.sorted_by_value_tuples.remove((old_value, key))

        self.dict[key] = new_value
        self.sorted_by_value_tuples.add((new_value, key))

    def pop_max(self):
        value, key = self.sorted_by_value_tuples.pop(-1)
        del self.dict[key]
        return key, value
    
    def pop_random(self):
        value, key = self.sorted_by_value_tuples.pop(random.randint(0, len(self.sorted_by_value_tuples)-1))
        del self.dict[key]
        return key, value

    def __contains__(self, key):
        return key in self.dict

    def __len__(self):
        return len(self.dict)



class CustomKNN():
    def __init__(self, k, class_negative_value, class_positive_value):
        self.k = k
        self.points_needed_for_class = math.ceil(self.k/2)
        self.class_negative_value = class_negative_value
        self.class_positive_value = class_positive_value
        self.num_threads = os.cpu_count()

    def fit(self, X, y):
        self.x_train = np.array(X)
        self.y_train = np.array(y)

        self.train_points_tree = KDTree(self.x_train, metric='euclidean')

    def predict(self, X):
        self.predictions = [None] * len(X)
        
            # Compute Euclidean distances
            # distances = np.linalg.norm(self.x_train - x, axis=1)
            # min_heap = [(dist, idx) for idx, dist in enumerate(distances)]
            # heapq.heapify(min_heap)
            
            # last_dist, idx = heapq.heappop(min_heap)
            # selected_pts = 0
            # possibly_pre_kth_dist_pts = 1
            # pos_num = int(self.y_train[idx] == self.class_positive_value)
            # while min_heap and ((selected_pts + possibly_pre_kth_dist_pts) != self.k):
            #     dist, idx = heapq.heappop(min_heap)

            #     pos_num += int(self.y_train[idx] == self.class_positive_value)

            #     if dist > last_dist:
            #         selected_pts += possibly_pre_kth_dist_pts
            #         possibly_pre_kth_dist_pts = 0
            #         last_dist = dist
            #     possibly_pre_kth_dist_pts += 1
                
            # pos_num -= int(self.y_train[idx] == self.class_positive_value)
            # while min_heap and (last_dist == dist) and (selected_pts != self.k):

            #     if self.y_train[idx] == self.class_positive_value:
            #         pos_num += 1
            #         selected_pts += 1
            #     dist, idx = heapq.heappop(min_heap)

        futures = []
        di, mo = divmod(len(X), self.num_threads)
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i in range(self.num_threads):
                start = i * di + min(i, mo)
                end = (i + 1) * di + min(i + 1, mo)
                futures.append(executor.submit(self._points_predict_process, range(start, end), X[start:end]))

            # Wait for all threads to complete
            for future in as_completed(futures):
                future.result()  # This will raise exceptions if any occurred
        return np.array(self.predictions)
                
    def _points_predict_process(self, indices, X):
        for i, x in zip(indices, X):

            # distances = np.linalg.norm(self.x_train - x, axis=1)
            # kth_value = self._quickselect(distances, self.k)

            # neighbors = 0
            # pos_num = 0
            # kth_dist_pos_num = 0
            # for j, distance in enumerate(distances):
            #     if distance < kth_value:
            #         pos_num += int(self.y_train[j] == self.class_positive_value)
            #         neighbors += 1
            #     elif distance == kth_value:
            #         kth_dist_pos_num += int(self.y_train[j] == self.class_positive_value)

            # while (neighbors < self.k) and (kth_dist_pos_num > 0):
            #     neighbors += 1
            #     kth_dist_pos_num -= 1

            # if pos_num >= self.points_needed_for_class:
            #     self.predictions[i] = self.class_positive_value
            # else:
            #     self.predictions[i] = self.class_negative_value 



            # sorted_distances = np.linalg.norm(self.x_train - x, axis=1)
            # sorted_indices = np.argsort(sorted_distances)
            # sorted_distances = sorted_distances[sorted_indices]

            # # Find distance of the k-th neighbor
            # check_kth_dist_index = self.k - 1
            # kth_dist = sorted_distances[check_kth_dist_index]

            # # Find which points are not at the k-th distance and count the positives
            # while (check_kth_dist_index > 0) and (sorted_distances[check_kth_dist_index - 1] == kth_dist):
            #     check_kth_dist_index -= 1
            # not_kth_dist_indices = sorted_indices[:check_kth_dist_index]
            # pos_num = (self.y_train[not_kth_dist_indices] == self.class_positive_value).sum()

            # # Get positive points at the k-th distance until there are k or the distance increases
            # points_needed = self.k - check_kth_dist_index
            # while (points_needed > 0) and (check_kth_dist_index < len(sorted_indices)) and (sorted_distances[check_kth_dist_index] == kth_dist):
            #     if self.y_train[sorted_indices[check_kth_dist_index]] == self.class_positive_value:
            #         points_needed -= 1
            #         pos_num += 1
            #     check_kth_dist_index += 1

            # if pos_num >= self.points_needed_for_class:
            #     self.predictions[i] = self.class_positive_value
            # else:
            #     self.predictions[i] = self.class_negative_value



            # Get the distances of the k neighbors
            dists, _ = self.train_points_tree.query([x], k=self.k, sort_results=True)
            # Get the indices of training points within the distance of the kth point
            row_indices, all_dists = self.train_points_tree.query_radius([x], r=dists[0][-1] + 1e-8, return_distance=True)

            # count positive neighbors, preferring positive ones when there are ties
            kth_dist_indices = all_dists[0] == dists[0][-1]
            pre_kth_num = np.sum(~kth_dist_indices)
            pre_kth_pos_num = np.sum((self.y_train[row_indices[0]][~kth_dist_indices] == self.class_positive_value))
            kth_pos_num = np.sum((self.y_train[row_indices[0]][kth_dist_indices] == self.class_positive_value))
            pos_num = min(self.k - pre_kth_num, kth_pos_num) + pre_kth_pos_num

            if pos_num >= self.points_needed_for_class:
                self.predictions[i] = self.class_positive_value
            else:
                self.predictions[i] = self.class_negative_value

def knn_cross_val_for_k_values(df,
                            values_to_check,
                            k_value,
                            sensitive_attr,
                            dominant_class_value,
                            class_attr,
                            class_negative_value,
                            class_positive_value,
                            exclude_sensitive_attr,
                            folds_num,
                            val_split,
                            val_results_path,
                            test_results_path,
                            scv_folder_extra_name,
                            check_random_removals_equal_to_each_fold,
                            random_removals_experiment_to_check_path,
                            removed_ids_experiment_extra_file_name):
    
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=42)

    possible_ids_to_remove_path = val_results_path + "possible_ids_to_remove/"

    if check_random_removals_equal_to_each_fold:
        val_cross_val_results_path = val_results_path + random_removals_experiment_to_check_path + scv_folder_extra_name + str(folds_num) + "_fold_scv"
        test_cross_val_results_path = test_results_path + random_removals_experiment_to_check_path + scv_folder_extra_name + str(folds_num) + "_fold_scv"
    else:
        val_cross_val_results_path = val_results_path + scv_folder_extra_name + str(folds_num) + "_fold_scv"
        test_cross_val_results_path = test_results_path + scv_folder_extra_name + str(folds_num) + "_fold_scv"
    if not exclude_sensitive_attr:
        val_cross_val_results_path += "_with_sens_attr"
        test_cross_val_results_path += "_with_sens_attr"
    if not os.path.exists(val_cross_val_results_path):
        os.makedirs(val_cross_val_results_path)  
    if not os.path.exists(test_cross_val_results_path):
        os.makedirs(test_cross_val_results_path)
    val_cross_val_results_path += "/"
    test_cross_val_results_path += "/"

    for var in values_to_check:
        if k_value is not None:
            knn = CustomKNN(k=k_value, class_negative_value=class_negative_value, class_positive_value=class_positive_value)
        else:
            knn = CustomKNN(k=var, class_negative_value=class_negative_value, class_positive_value=class_positive_value)

        print(f"\nRunning for var={var}")
        avg_val_dom_pos_percent = avg_val_sens_pos_percent = avg_val_dataset_acc = avg_val_pos_prec = avg_val_pos_rec = avg_val_neg_prec = avg_val_neg_rec = 0
        avg_test_dom_pos_percent = avg_test_sens_pos_percent = avg_test_dataset_acc = avg_test_pos_prec = avg_test_pos_rec = avg_test_neg_prec = avg_test_neg_rec = 0
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(df.drop(class_attr, axis=1), df[class_attr]), 1):
            print(f"\nFold {fold}")
            x_train, x_val, y_train, dataset_y_val = train_test_split(df.drop(class_attr, axis=1).iloc[train_val_idx], 
                                                                    df[class_attr].iloc[train_val_idx], 
                                                                    test_size=val_split, 
                                                                    stratify=df[class_attr].iloc[train_val_idx],
                                                                    random_state=42) 
            x_test, dataset_y_test = df.drop(class_attr, axis=1).iloc[test_idx], df[class_attr].iloc[test_idx]

            if check_random_removals_equal_to_each_fold:
                random.seed(345)
                if k_value is not None:
                    possible_train_ids_to_remove_per_neg_val_point_df = pd.read_csv(possible_ids_to_remove_path + "fold_" + str(fold) + "/" + str(k_value) + "_neigh_possible_train_ids_to_remove_per_neg_val_point.txt", delimiter=":")
                else:
                    possible_train_ids_to_remove_per_neg_val_point_df = pd.read_csv(possible_ids_to_remove_path + "fold_" + str(fold) + "/" + str(var) + "_neigh_possible_train_ids_to_remove_per_neg_val_point.txt", delimiter=":")

                possible_ids_to_remove = set()
                for _, row in possible_train_ids_to_remove_per_neg_val_point_df.iterrows():
                    possible_ids_to_remove.update(ast.literal_eval(row.iloc[1]))
                train_ids_removed_df = pd.read_csv(val_results_path + "/" + random_removals_experiment_to_check_path + "train_ids_removed/fold_" + str(fold) + "/" + str(var) + removed_ids_experiment_extra_file_name + "train_ids_removed.txt", header=None)
                train_ids_to_remove = random.sample(list(possible_ids_to_remove), len(train_ids_removed_df))
                x_train = x_train.drop(index=train_ids_to_remove)
                y_train = y_train.drop(index=train_ids_to_remove)

            # train_dom_mask = x_train[sensitive_attr] == dominant_class_value
            val_dom_mask = x_val[sensitive_attr] == dominant_class_value
            test_dom_mask = x_test[sensitive_attr] == dominant_class_value

            if exclude_sensitive_attr:
                x_train = x_train.drop(sensitive_attr, axis=1)
                x_val = x_val.drop(sensitive_attr, axis=1)
                x_test = x_test.drop(sensitive_attr, axis=1)
            
            knn.fit(x_train.values, y_train.values)

            print(f"\nTraining ids: {x_train.shape[0]}")
            print(f"Testing ids: {x_test.shape[0]}")
            print(f"Validation ids: {x_val.shape[0]}")

            fold_val_predictions = pd.Series(knn.predict(x_val.values), index=x_val.index)
            fold_test_predictions = pd.Series(knn.predict(x_test.values), index=x_test.index)

            avg_val_dom_pos_percent += ((fold_val_predictions[val_dom_mask] == class_positive_value).sum()/val_dom_mask.sum())/folds_num
            avg_val_sens_pos_percent += ((fold_val_predictions[~val_dom_mask] == class_positive_value).sum()/(~val_dom_mask).sum())/folds_num
            avg_val_dataset_acc += (accuracy_score(dataset_y_val, fold_val_predictions))/folds_num
            avg_val_pos_prec += (precision_score(dataset_y_val, fold_val_predictions, pos_label=class_positive_value))/folds_num
            avg_val_pos_rec += (recall_score(dataset_y_val, fold_val_predictions, pos_label=class_positive_value))/folds_num
            avg_val_neg_prec += (precision_score(dataset_y_val, fold_val_predictions, pos_label=class_negative_value))/folds_num
            avg_val_neg_rec += (recall_score(dataset_y_val, fold_val_predictions, pos_label=class_negative_value))/folds_num

            avg_test_dom_pos_percent += ((fold_test_predictions[test_dom_mask] == class_positive_value).sum()/test_dom_mask.sum())/folds_num
            avg_test_sens_pos_percent += ((fold_test_predictions[~test_dom_mask] == class_positive_value).sum()/(~test_dom_mask).sum())/folds_num
            avg_test_dataset_acc += (accuracy_score(dataset_y_test, fold_test_predictions))/folds_num
            avg_test_pos_prec += (precision_score(dataset_y_test, fold_test_predictions, pos_label=class_positive_value))/folds_num
            avg_test_pos_rec += (recall_score(dataset_y_test, fold_test_predictions, pos_label=class_positive_value))/folds_num
            avg_test_neg_prec += (precision_score(dataset_y_test, fold_test_predictions, pos_label=class_negative_value))/folds_num
            avg_test_neg_rec += (recall_score(dataset_y_test, fold_test_predictions, pos_label=class_negative_value))/folds_num


        val_cross_val_results_file_name = val_cross_val_results_path + str(var) + removed_ids_experiment_extra_file_name + str(folds_num) + "_folds_avg_val"
        test_cross_val_results__file_name = test_cross_val_results_path + str(var) + removed_ids_experiment_extra_file_name + str(folds_num) + "_folds_avg_test"
        if not exclude_sensitive_attr:
            val_cross_val_results_file_name += "_with_sens_attr"
            test_cross_val_results__file_name += "_with_sens_attr"
        val_cross_val_results_file_name += ".txt"
        test_cross_val_results__file_name += ".txt"

        with open(val_cross_val_results_file_name, "w") as f1, \
            open(test_cross_val_results__file_name, "w") as f2:
        
            f1.write("avg_val_dom_pos_percent:" + str(avg_val_dom_pos_percent) + "\n")
            f1.write("avg_val_sens_pos_percent:" + str(avg_val_sens_pos_percent) + "\n")
            f1.write("avg_val_dataset_acc:" + str(avg_val_dataset_acc) + "\n")
            f1.write("avg_val_pos_prec:" + str(avg_val_pos_prec) + "\n")
            f1.write("avg_val_pos_rec:" + str(avg_val_pos_rec) + "\n")
            f1.write("avg_val_neg_prec:" + str(avg_val_neg_prec) + "\n")
            f1.write("avg_val_neg_rec:" + str(avg_val_neg_rec) + "\n")

            f2.write("avg_test_dom_pos_percent:" + str(avg_test_dom_pos_percent) + "\n")
            f2.write("avg_test_sens_pos_percent:" + str(avg_test_sens_pos_percent) + "\n")
            f2.write("avg_test_dataset_acc:" + str(avg_test_dataset_acc) + "\n")
            f2.write("avg_test_pos_prec:" + str(avg_test_pos_prec) + "\n")
            f2.write("avg_test_pos_rec:" + str(avg_test_pos_rec) + "\n")
            f2.write("avg_test_neg_prec:" + str(avg_test_neg_prec) + "\n")
            f2.write("avg_test_neg_rec:" + str(avg_test_neg_rec) + "\n")

        print(df.describe(include="all"))

    

        
if __name__ == "__main__":

    # k_values_to_check = range(9, 101)
    # class_attr = "class"
    # class_negative_value = 2
    # class_positive_value = 1
    # sensitive_attr = "Sex"
    # dominant_class_value = "male"
    # sensitive_class_value = "female"
    # exclude_sensitive_attr = True
    # load_from = "data/dutch_cencus_wtarget.csv"
    # results_folder_name = os.path.splitext(os.path.basename(load_from))[0]
    # val_results_path = "results/" + results_folder_name + "/val_results/"
    # test_results_path = "results/" + results_folder_name + "/test_results/"
    # folds_num = 10
    # val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds
    # scv_folder_extra_name = "knn_"

    # df = pd.read_csv(load_from)


    # k_values_to_check = range(1, 50, 2)
    # percentage_difs = [None] # np.arange(0.015, 0.0020, -0.0025)
    # sensitive_increases = [0.25] # np.arange(0.05, 0.55, 0.05) # 
    # class_attr = "Income"
    # class_negative_value = "Negative Class\n(< 50000)"
    # class_positive_value = "Positive Class\n(>= 50000)"
    # sensitive_attr = "Gender"
    # dominant_class_value = "Male"
    # sensitive_class_value = "Female"
    # exclude_sensitive_attr = True
    # save_folder_name = "acs_income_gender"
    # val_results_path = "results/" + save_folder_name + "/val_results/"
    # test_results_path = "results/" + save_folder_name + "/test_results/"
    # folds_num = 10
    # val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds
    # check_random_removals = True          # otherwise evaluates normal knn
    # random_removals_experiment_path = "10_fold_scv/over_k/sens_incr_0.25/" # "10_fold_scv/over_k/perc_dif_0.001/" # "10_fold_scv/over_sens_incr/21_neigh/"

    # df = datasets.fetch_acs_income()["data"]
    # df["Income"] = datasets.fetch_acs_income()["target"]
    # df = df.sample(n=50000, random_state=42)
    # df = df.reset_index(drop=True)
    # df["Income"] = df["Income"].apply(lambda x: "Positive Class\n(>= 50000)" if x>=50000 else "Negative Class\n(< 50000)")
    # df["Gender"] = df["SEX"].apply(lambda x: "Male" if x==1.0 else "Female")
    # df = df.drop(columns=["SEX"])


    # k_values_to_check = range(1, 50, 2)
    # percentage_difs = [0.001] # np.arange(0.015, 0.0020, -0.0025)
    # sensitive_increases = [0.25] # np.arange(0.05, 0.55, 0.05)
    # class_attr = "Income"
    # class_negative_value = "Less than 50000"
    # class_positive_value = "Greater or equal to 50000"
    # sensitive_attr = "Race"
    # dominant_class_value = "White"
    # sensitive_class_value = "Non-white"
    # exclude_sensitive_attr = True
    # results_folder_name = "acs_income_race"
    # val_results_path = "results/" + results_folder_name + "/val_results/"
    # test_results_path = "results/" + results_folder_name + "/test_results/"
    # folds_num = 10
    # val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds
    # scv_folder_extra_name = "random_removals_"
    # check_random_removals_equal_to_each_fold = False   # otherwise evaluates normal knn

    # df = datasets.fetch_acs_income()["data"]
    # df["Income"] = datasets.fetch_acs_income()["target"]
    # df = df.sample(n=50000, random_state=42)
    # df = df.reset_index(drop=True)
    # df["Income"] = df["Income"].apply(lambda x: "Greater or equal to 50000" if x>=50000 else "Less than 50000")
    # df["Race"] = df["RAC1P"].apply(lambda x:"White" if x==1.0 else "Non-white")
    # df = df.drop(columns=["RAC1P"])


    # k_values_to_check = range(1, 50, 2)
    # percentage_difs = [0.001] # np.arange(0.015, 0.0020, -0.0025)
    # sensitive_increases = [0.25] # np.arange(0.05, 0.55, 0.05)
    # class_attr = "Income"
    # class_negative_value = 0
    # class_positive_value = 1
    # sensitive_attr = "Gender"
    # dominant_class_value = "Male"
    # sensitive_class_value = "Female"
    # exclude_sensitive_attr = True
    # results_folder_name = "adult_gender"
    # val_results_path = "results/" + results_folder_name + "/val_results/"
    # test_results_path = "results/" + results_folder_name + "/test_results/"
    # folds_num = 10
    # val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds
    # check_random_removals_equal_to_each_fold = False   # otherwise evaluates normal knn

    # df = fetch_ucirepo(id=2).data.features
    # df = df.rename(columns={"sex": "Gender"})
    # df["Income"] = fetch_ucirepo(id=2).data.targets
    # df["Income"] = ((df["Income"] == ">50K") | (df["Income"] == ">50K.")).astype(int)
    # df = df.drop(columns = ["education"])


    # k_values_to_check = range(1, 50, 2)
    # percentage_difs = [None] # np.arange(0.015, 0.0020, -0.0025)
    # sensitive_increases = np.arange(0.05, 0.55, 0.05) # [0.25] 
    # class_attr = "Income"
    # class_negative_value = 0
    # class_positive_value = 1
    # sensitive_attr = "Race"
    # dominant_class_value = "White"
    # sensitive_class_value = "Other"
    # exclude_sensitive_attr = True
    # results_folder_name = "adult_race"
    # val_results_path = "results/" + results_folder_name + "/val_results/"
    # test_results_path = "results/" + results_folder_name + "/test_results/"
    # folds_num = 10
    # val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds
    # check_random_removals_equal_to_each_fold = False          # otherwise evaluates normal knn

    # df = fetch_ucirepo(id=2).data.features
    # df = df.rename(columns={"race": "Race"})
    # df["Income"] = fetch_ucirepo(id=2).data.targets
    # df["Income"] = ((df["Income"] == ">50K") | (df["Income"] == ">50K.")).astype(int)
    # df = df.drop(columns = ["education"])
    # df["Race"] = df["Race"].apply(lambda x: x if x=="White" else "Other")


    k_values_to_check = range(1, 50, 2)
    percentage_difs = [None] # [0.001] # np.arange(0.015, 0.0020, -0.0025)
    sensitive_increases = [0.25] # np.arange(0.05, 0.55, 0.05)
    class_attr = "Default payment"
    class_negative_value = "No"
    class_positive_value = "Yes"
    sensitive_attr = "Education"
    dominant_class_value = "University or High School"
    sensitive_class_value = "Graduate School or Others"
    exclude_sensitive_attr = True
    save_folder_name = "credit_card_default"
    val_results_path = "results/" + save_folder_name + "/val_results/"
    test_results_path = "results/" + save_folder_name + "/test_results/"
    folds_num = 10
    val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds
    check_random_removals_equal_to_each_fold = False          # otherwise evaluates normal knn

    df = fetch_ucirepo(id=350).data.features
    df["Default payment"] = fetch_ucirepo(id=350).data.targets
    df["Default payment"] = df["Default payment"].apply(lambda x: "Yes" if x==1 else "No")
    df["Education"] = df["X3"].apply(lambda x: "University or High School" if (x==2 or x==3) else "Graduate School or Others")
    df = df.drop(columns=["X3"])

    # k_values_to_check = range(1, 50, 2)
    # percentage_difs = [None] # np.arange(0.015, 0.0020, -0.0025)
    # sensitive_increases = np.arange(0.05, 0.55, 0.05) # [0.25] 
    # class_attr = "Subscribed"
    # class_negative_value = "no"
    # class_positive_value = "yes"
    # sensitive_attr = "Marital"
    # dominant_class_value = "Divorced or single"
    # sensitive_class_value = "Married"
    # exclude_sensitive_attr = True
    # save_folder_name = "bank_marketing"
    # val_results_path = "results/" + save_folder_name + "/val_results/"
    # test_results_path = "results/" + save_folder_name + "/test_results/"
    # folds_num = 10
    # val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds
    # check_random_removals_equal_to_each_fold = False          # otherwise evaluates normal knn

    # df = fetch_ucirepo(id=222).data.features
    # df = df.rename(columns={"marital": "Marital"})
    # df["Subscribed"] = fetch_ucirepo(id=222).data.targets
    # df["Marital"] = df["Marital"].apply(lambda x: "Married" if x=="married" else "Divorced or single")


    if not check_random_removals_equal_to_each_fold:
        scv_folder_extra_name = "knn_"
    else:
        scv_folder_extra_name = "random_removals_equal_to_alg_folds"

    if len(k_values_to_check) > 1:
        removed_ids_experiment_extra_file_name = "_neigh_"
        values_to_check = k_values_to_check
        k_value = None
        if percentage_difs[0] is not None:
            random_removals_experiment_to_check_path = "10_fold_scv/over_k/perc_dif_" + str(percentage_difs[0]) + "/"
        else:
            random_removals_experiment_to_check_path = "10_fold_scv/over_k/sens_incr_" + str(sensitive_increases[0]) + "/"
    elif percentage_difs[0] is not None:
        removed_ids_experiment_extra_file_name = "_perc_dif_"
        values_to_check = percentage_difs
        k_value = k_values_to_check[0]
        random_removals_experiment_to_check_path = "10_fold_scv/over_perc_dif/" + str(k_value) + "_neigh/"
    else:
        removed_ids_experiment_extra_file_name = "_sens_incr_"
        values_to_check = sensitive_increases
        k_value = k_values_to_check[0]
        random_removals_experiment_to_check_path = "10_fold_scv/over_sens_incr/" + str(k_value) + "_neigh/"

    dominant_class_value, sensitive_class_value, class_negative_value, class_positive_value = encode_dataframe(df, sensitive_attr, dominant_class_value, sensitive_class_value, class_attr, class_negative_value, class_positive_value)
    knn_cross_val_for_k_values(df=df,
                        values_to_check=values_to_check,
                        k_value=k_value,
                        class_attr=class_attr,
                        class_negative_value=class_negative_value,
                        class_positive_value=class_positive_value,
                        sensitive_attr=sensitive_attr,
                        dominant_class_value=dominant_class_value,
                        exclude_sensitive_attr=exclude_sensitive_attr,
                        val_split=val_split,
                        val_results_path=val_results_path,
                        test_results_path=test_results_path,
                        scv_folder_extra_name=scv_folder_extra_name,
                        folds_num=folds_num,
                        check_random_removals_equal_to_each_fold=check_random_removals_equal_to_each_fold,
                        random_removals_experiment_to_check_path=random_removals_experiment_to_check_path,
                        removed_ids_experiment_extra_file_name=removed_ids_experiment_extra_file_name)