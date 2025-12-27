import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, KNeighborsClassifier
from collections import defaultdict
from utils.data_structures import ValueSortedDict, CustomKNN
from utils.utilities import encode_dataframe, save_dataset_pos_rates
from utils.graphs import plot_dataset_stats
from sklearn.metrics import accuracy_score, precision_score, recall_score
import math
import os
from sklearn.model_selection import StratifiedKFold
from fairlearn import datasets
from ucimlrepo import fetch_ucirepo



class FairKnnWithRemovals:
    def __init__(self,
                 k,
                 class_attr,
                 class_negative_value,
                 class_positive_value,
                 sensitive_attr,
                 dominant_class_value,
                 sensitive_class_value,
                 exlude_sensitive_attr,
                 save_results,
                 metrics_recalculate_gap,
                 val_results_path,
                 test_results_path,
                 percentage_dif,
                 sensitive_increase,
                 check_random_removals_until_fair,
                 x_train,
                 x_val,
                 x_test,
                 y_train,
                 dataset_y_val,
                 dataset_y_test,
                 removed_ids_save_folder_path,
                 possible_and_removed_ids_extra_folder_name,
                 removed_ids_extra_file_name):
        self.k = k
        self.max_neg_neighbors_for_pos = math.floor(self.k/2)
        self.class_attr = class_attr
        self.class_negative_value = class_negative_value
        self.class_positive_value = class_positive_value
        self.sensitive_attr = sensitive_attr
        self.dominant_class_value = dominant_class_value
        self.sensitive_class_value = sensitive_class_value
        self.exclude_sensitive_attr = exlude_sensitive_attr
        self.percentage_dif = percentage_dif
        self.sensitive_increase = sensitive_increase
        self.check_random_removals_until_fair = check_random_removals_until_fair
        self.val_results_path = val_results_path
        self.save_results = save_results
        self.metrics_recalculate_gap = metrics_recalculate_gap
        self.test_results_path = test_results_path
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.dataset_y_val = dataset_y_val
        self.dataset_y_test = dataset_y_test
        self.removed_ids_save_folder_path = removed_ids_save_folder_path
        self.possible_and_removed_ids_extra_folder_name = str(possible_and_removed_ids_extra_folder_name)
        self.removed_ids_extra_file_name = removed_ids_extra_file_name


        self.model = CustomKNN(k=self.k, class_negative_value=self.class_negative_value, class_positive_value=self.class_positive_value)
        # self.model = KNeighborsClassifier(n_neighbors=self.k)

        self.train_dom_mask = self.x_train[self.sensitive_attr] == self.dominant_class_value
        self.val_dom_mask = self.x_val[self.sensitive_attr] == self.dominant_class_value
        self.test_dom_mask = self.x_test[self.sensitive_attr] == self.dominant_class_value

        if self.exclude_sensitive_attr:
            self.x_train = self.x_train.drop(self.sensitive_attr, axis=1)
            self.x_val = self.x_val.drop(self.sensitive_attr, axis=1)
            self.x_test = self.x_test.drop(self.sensitive_attr, axis=1)

        print("Statistics before running the algorithm or predicting(dataset stats):\n")
        self._print_dataset_stats() # original statistics from the dataset

        # dict with keys negative validation ids and values the negative training ids sets to remove to become positive
        self.D_v = {}

        # dicts with keys negative training ids and values negative validation ids they affect (inverse of D_v)
        # N_d is for the dominant class training ids while N_s is for the sensitive class
        self.N_d = defaultdict(set)
        self.N_s = defaultdict(set)

        # value-sorted dict with keys negative training ids and values their weights
        self.weight_sorted_dict = ValueSortedDict()

    def fit_predict_metrics(self):
        
        self.model.fit(self.x_train.values, self.y_train.values)
        self.first_pred_y_val = pd.Series(self.model.predict(self.x_val.values), index = self.x_val.index)
        self.first_pred_y_test = pd.Series(self.model.predict(self.x_test.values), index = self.x_test.index)
        print("\nFirst predictions done")

        self._prepare_structures()
        print("\nStructures ready\n")

        self.val_pred_dom_pos_count = (self.first_pred_y_val[self.val_dom_mask] == self.class_positive_value).sum()
        self.val_pred_sens_pos_count = (self.first_pred_y_val[~self.val_dom_mask] == self.class_positive_value).sum()
        self.val_dom_count = self.val_dom_mask.sum()
        self.val_sens_count = (~self.val_dom_mask).sum()
        self.first_pred_val_sens_pos_percent = self.val_pred_sens_pos_count / self.val_sens_count

        starting_train_points_num = len(self.y_train)

        num_of_ids_removed = -1
        if self.save_results:

            if not os.path.exists(self.val_results_path):
                os.makedirs(self.val_results_path)

            if not os.path.exists(self.test_results_path):
                os.makedirs(self.test_results_path)

            if self.exclude_sensitive_attr:
                with open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dom_pos_perc.txt", "w") as f1, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_sens_pos_perc.txt", "w") as f2, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_first_pred_acc.txt", "w") as f3, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_acc.txt", "w") as f4, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_pos_class_prec.txt", "w") as f5, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_pos_class_rec.txt", "w") as f6, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_neg_class_prec.txt", "w") as f7, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_neg_class_rec.txt", "w") as f8, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dom_pos_perc.txt", "w") as f9, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_sens_pos_perc.txt", "w") as f10, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_first_pred_acc.txt", "w") as f11, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_acc.txt", "w") as f12, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_pos_class_prec.txt", "w") as f13, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_pos_class_rec.txt", "w") as f14, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_neg_class_prec.txt", "w") as f15, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_neg_class_rec.txt", "w") as f16:
                    num_of_ids_removed = self._run_algorithm(files_to_save=[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16])
            else:
                with open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dom_pos_perc_with_sens_attr.txt", "w") as f1, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_sens_pos_perc_with_sens_attr.txt", "w") as f2, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_first_pred_acc_with_sens_attr.txt", "w") as f3, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_acc_with_sens_attr.txt", "w") as f4, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_pos_class_prec_with_sens_attr.txt", "w") as f5, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_pos_class_rec_with_sens_attr.txt", "w") as f6, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_neg_class_prec_with_sens_attr.txt", "w") as f7, \
                     open(self.val_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_val_dataset_neg_class_rec_with_sens_attr.txt", "w") as f8, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dom_pos_perc_with_sens_attr.txt", "w") as f9, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_sens_pos_perc_with_sens_attr.txt", "w") as f10, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_first_pred_acc_with_sens_attr.txt", "w") as f11, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_acc_with_sens_attr.txt", "w") as f12, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_pos_class_prec_with_sens_attr.txt", "w") as f13, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_pos_class_rec_with_sens_attr.txt", "w") as f14, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_neg_class_prec_with_sens_attr.txt", "w") as f15, \
                     open(self.test_results_path + str(self.k) + "_neigh_" + str(self.metrics_recalculate_gap) + "_gap_test_dataset_neg_class_rec_with_sens_attr.txt", "w") as f16:
                    num_of_ids_removed = self._run_algorithm(files_to_save=[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16])
        else:
            num_of_ids_removed = self._run_algorithm(files_to_save=False)
        
        self.model.fit(self.x_train.values, self.y_train.values)
        val_results = self._val_from_model(files_to_save=False)
        val_results.update({"train_ids_removed": num_of_ids_removed, "percentage_of_train_ids_removed": num_of_ids_removed/starting_train_points_num})
        return {"validation": val_results, "test": self._test_from_model(files_to_save=False)}
    
    def _run_algorithm(self, files_to_save):
        print("\nStatistics after the first prediction but before running the algorithm:\n")
        is_fair, val_pred_dom_pos_percent, val_pred_sens_pos_percent = self._is_fair()
        print("Dominant class positive percentage for validation set:", val_pred_dom_pos_percent)
        print("Sensitive class positive percentage for validation set:", val_pred_sens_pos_percent)
        self._val_from_model(files_to_save=files_to_save)
        self._test_from_model(files_to_save=files_to_save)
        
        affected_train_ids_set = set()
        if self.check_random_removals_until_fair:
            random.seed(345)
            get_next_train_id = self.weight_sorted_dict.pop_random
            update_affected_ids_set = lambda x: None
        else:
            get_next_train_id = self.weight_sorted_dict.pop_max
            update_affected_ids_set = affected_train_ids_set.update

        iteration = 0
        train_ids_removed_path = self.removed_ids_save_folder_path + "train_ids_removed/" + self.possible_and_removed_ids_extra_folder_name + "/"
        if not os.path.exists(train_ids_removed_path):
            os.makedirs(train_ids_removed_path)
        with open(train_ids_removed_path + self.removed_ids_extra_file_name + "train_ids_removed.txt", "w") as f:
            while not is_fair:
                
                # all train ids have the same weights with random removal
                train_id_to_remove, _ = get_next_train_id()
                f.write(str(train_id_to_remove)+"\n")

                self.x_train.drop(index = train_id_to_remove, inplace=True)
                self.y_train.drop(index = train_id_to_remove, inplace=True)

                affected_train_ids_set = set()
                for val_id in self.N_d[train_id_to_remove]:
                    self.D_v[val_id].remove(train_id_to_remove)
                    if len(self.D_v[val_id]) <= self.max_neg_neighbors_for_pos:
                        # updating other training points because this val point is already positive
                        for train_id in self.D_v[val_id]:
                            self.N_d[train_id].remove(val_id)
                        del self.D_v[val_id]
                        self.val_pred_dom_pos_count += 1
                    # else:
                    #     affected_train_ids_set.update(self.D_v[val_id])
                del self.N_d[train_id_to_remove]

                for val_id in self.N_s[train_id_to_remove]:
                    self.D_v[val_id].remove(train_id_to_remove)
                    if len(self.D_v[val_id]) <= self.max_neg_neighbors_for_pos:
                        # updating other training points because this val point is already positive
                        for train_id in self.D_v[val_id]:
                            self.N_s[train_id].remove(val_id)
                        del self.D_v[val_id]
                        self.val_pred_sens_pos_count += 1
                    else:
                        update_affected_ids_set(self.D_v[val_id])
                del self.N_s[train_id_to_remove]

                for affected_train_id in affected_train_ids_set:
                    self.weight_sorted_dict.update(affected_train_id, self._calculate_weight(affected_train_id))

                iteration += 1
                # print(f"Finished iteration {iteration}")
                is_fair, val_pred_dom_pos_percent, val_pred_sens_pos_percent = self._is_fair()
                if iteration%self.metrics_recalculate_gap == 0:
                    
                    print(f"\nStatistics after iteration {iteration}:\n")
                    print(f"Removed train id {train_id_to_remove}")
                    print("Dominant class positive percentage for validation set:", val_pred_dom_pos_percent)
                    print("Sensitive class positive percentage for validation set:", val_pred_sens_pos_percent)
                    self.model.fit(self.x_train.values, self.y_train.values)
                    self._val_from_model(files_to_save=files_to_save)
                    self._test_from_model(files_to_save=files_to_save)
        
        print(f"\nValidation set is now fair after {iteration} iterations")
        print("\nFinal results\n")
        print("Dominant class positive percentage for validation set:", val_pred_dom_pos_percent)
        print("Sensitive class positive percentage for validation set:", val_pred_sens_pos_percent)

        return iteration

    def _is_fair(self):
        val_pred_dom_pos_percent = self.val_pred_dom_pos_count / self.val_dom_count
        val_pred_sens_pos_percent = self.val_pred_sens_pos_count / self.val_sens_count
        
        if self.percentage_dif:
            return [(val_pred_dom_pos_percent - val_pred_sens_pos_percent) <= self.percentage_dif, val_pred_dom_pos_percent, val_pred_sens_pos_percent]
        else:
            return [(val_pred_sens_pos_percent - self.first_pred_val_sens_pos_percent) >= self.sensitive_increase, val_pred_dom_pos_percent, val_pred_sens_pos_percent]

    def _prepare_structures(self):

        positive_train_points_tree = KDTree(self.x_train[self.y_train == self.class_positive_value].values, metric='euclidean')
        negative_train_points_tree = KDTree(self.x_train[~(self.y_train == self.class_positive_value)].values, metric='euclidean')

        possible_ids_to_remove_path = self.val_results_path + "possible_ids_to_remove/" + self.possible_and_removed_ids_extra_folder_name + "/"
        if not os.path.exists(possible_ids_to_remove_path):
            os.makedirs(possible_ids_to_remove_path)
        with open(possible_ids_to_remove_path + str(self.k) + "_neigh_" + "possible_train_ids_to_remove_per_neg_val_point.txt", "w") as f:
            f.write("val_id:possible_neg_train_ids_to_be_removed:num_to_remove\n")
            # for each negative validation id
            for val_id, val_point in self.x_val.iterrows():
                if self.first_pred_y_val.loc[val_id] == self.class_negative_value:

                    self.D_v[val_id] = self._get_nearest_neg_training_neighbors_set_to_become_pos(val_point, positive_train_points_tree, negative_train_points_tree)
                    f.write(str(val_id) + ":" + str(self.D_v[val_id]) + ":" + str(len(self.D_v[val_id]) - self.max_neg_neighbors_for_pos) + "\n")

                    # if the point is from the dominant class
                    if self.val_dom_mask.loc[val_id] == True:
                        # for each negative training id in the set
                        for train_id in self.D_v[val_id]:
                            self.N_d[train_id].add(val_id)
                    # if the point is from the sensitive class
                    else:
                        # for each negative training id in the set
                        for train_id in self.D_v[val_id]:
                            self.N_s[train_id].add(val_id)

        del positive_train_points_tree, negative_train_points_tree

        if self.check_random_removals_until_fair:
            # for each negative training id that affects a sensitive negative validation point
            for train_id in self.N_s.keys():
                self.weight_sorted_dict.add(train_id, 1)
        else:
            # for each negative training id that affects a sensitive negative validation point
            for train_id in self.N_s.keys():
                self.weight_sorted_dict.add(train_id, self._calculate_weight(train_id))

    def _calculate_weight(self, train_id):
        
        # calculate sensitive class weight
        sw = 0
        for val_id in self.N_s[train_id]:
            sw += 1/len(self.D_v[val_id])

        if self.percentage_dif:
            # calculate dominant class weight
            dw = 0
            if train_id in self.N_d.keys():
                for val_id in self.N_d[train_id]:
                    dw += 1/len(self.D_v[val_id])
            return sw - dw
        else:
            return sw

    def _get_nearest_neg_training_neighbors_set_to_become_pos(self, val_point, positive_train_points_tree, negative_train_points_tree):
        
        # Get the sorted distances of the ceil(k/2) nearest positive training points
        pos_dists, _ = positive_train_points_tree.query([val_point.values], k=math.ceil(self.k/2), sort_results=True)
        # Get the indices of negative training points within the distance of the kth positive point
        neg_row_indices, neg_dists = negative_train_points_tree.query_radius([val_point.values], r=pos_dists[0][-1] + 1e-8, return_distance=True)
        # Remove negative points that are at the exact distance as the kth positive point
        neg_row_indices = [neg_row_indices[0][i] for i in range(len(neg_row_indices[0])) if neg_dists[0][i] != pos_dists[0][-1]]
        return set(self.x_train[~(self.y_train == self.class_positive_value)].iloc[neg_row_indices].index)

    def _val_from_model(self, files_to_save):
                
        y_val = pd.Series(self.model.predict(self.x_val.values), index = self.x_val.index)
        # print("Dominant validation ids changed: ", self.x_val[self.val_dom_mask].iloc[np.where((self.first_pred_y_val[self.val_dom_mask] == self.class_positive_value) != (y_val[self.val_dom_mask] == self.class_positive_value))[0]].index)
        # print("Sensitive validation ids changed: ", self.x_val[~self.val_dom_mask].iloc[np.where((self.first_pred_y_val[~self.val_dom_mask] == self.class_positive_value) != (y_val[~self.val_dom_mask] == self.class_positive_value))[0]].index)
        
        val_dom_pos_count = np.count_nonzero(y_val[self.val_dom_mask] == self.class_positive_value)
        val_dom_pos_percent = val_dom_pos_count/(self.val_dom_mask.sum())
        print("Manual dominant class positive percentage for validation set:", val_dom_pos_percent)
        val_sens_pos_count = np.count_nonzero(y_val[~self.val_dom_mask] == self.class_positive_value)
        val_sens_pos_percent = val_sens_pos_count/((~self.val_dom_mask).sum())
        print("Manual sensitive class positive percentage for validation set:", val_sens_pos_percent)
        
        val_first_pred_acc = accuracy_score(self.first_pred_y_val, y_val)
        print("Validation accuracy according to first prediction:", val_first_pred_acc)
        val_dataset_acc = accuracy_score(self.dataset_y_val, y_val)
        print("Validation accuracy according to dataset:", val_dataset_acc)
        val_pos_prec = precision_score(self.dataset_y_val, y_val, pos_label=self.class_positive_value)
        print("Validation positive class precision according to dataset:", val_pos_prec)
        val_pos_rec = recall_score(self.dataset_y_val, y_val, pos_label=self.class_positive_value)
        print("Validation positive class recall according to dataset:", val_pos_rec)
        val_neg_prec = precision_score(self.dataset_y_val, y_val, pos_label=self.class_negative_value)
        print("Validation negative class precision according to dataset:", val_neg_prec)
        val_neg_rec = recall_score(self.dataset_y_val, y_val, pos_label=self.class_negative_value)
        print("Validation negative class recall according to dataset:", val_neg_rec, "\n")
        
        if files_to_save:
            files_to_save[0].write(str(val_dom_pos_percent)+"\n")
            files_to_save[1].write(str(val_sens_pos_percent)+"\n")
            files_to_save[2].write(str(val_first_pred_acc)+"\n")
            files_to_save[3].write(str(val_dataset_acc)+"\n")
            files_to_save[4].write(str(val_pos_prec)+"\n")
            files_to_save[5].write(str(val_pos_rec)+"\n")
            files_to_save[6].write(str(val_neg_prec)+"\n")
            files_to_save[7].write(str(val_neg_rec)+"\n")
        
        return {"dom_pos_percent": val_dom_pos_percent, "sens_pos_percent": val_sens_pos_percent, "first_pred_acc": val_first_pred_acc, "dataset_acc": val_dataset_acc,
                "pos_prec": val_pos_prec, "pos_rec": val_pos_rec, "neg_prec": val_neg_prec, "neg_rec": val_neg_rec}

    def _test_from_model(self, files_to_save):
        
        y_test = self.model.predict(self.x_test.values)
        
        test_dom_pos_count = np.count_nonzero(y_test[self.test_dom_mask] == self.class_positive_value)
        test_dom_pos_percent = test_dom_pos_count/(self.test_dom_mask.sum())
        print("Manual dominant class positive percentage for testing set:", test_dom_pos_percent)
        test_sens_pos_count = np.count_nonzero(y_test[~self.test_dom_mask] == self.class_positive_value)
        test_sens_pos_percent = test_sens_pos_count/((~self.test_dom_mask).sum())
        print("Manual sensitive class positive percentage for testing set:", test_sens_pos_percent)

        test_first_pred_acc = accuracy_score(self.first_pred_y_test, y_test)
        print("Testing accuracy according to first prediction:", test_first_pred_acc)
        test_dataset_acc = accuracy_score(self.dataset_y_test, y_test)
        print("Testing accuracy according to dataset:", test_dataset_acc)
        test_pos_prec = precision_score(self.dataset_y_test, y_test, pos_label=self.class_positive_value)
        print("Testing positive class precision according to dataset:", test_pos_prec)
        test_pos_recall = recall_score(self.dataset_y_test, y_test, pos_label=self.class_positive_value)
        print("Testing positive class recall according to dataset:", test_pos_recall)
        test_neg_prec = precision_score(self.dataset_y_test, y_test, pos_label=self.class_negative_value)
        print("Testing negative class precision according to dataset:", test_neg_prec)
        test_neg_recall = recall_score(self.dataset_y_test, y_test, pos_label=self.class_negative_value)
        print("Testing negative class recall according to dataset:", test_neg_recall, "\n")

        if files_to_save:
            files_to_save[6].write(str(test_dom_pos_percent)+"\n")
            files_to_save[7].write(str(test_sens_pos_percent)+"\n")
            files_to_save[8].write(str(test_first_pred_acc)+"\n")
            files_to_save[9].write(str(test_dataset_acc)+"\n")
            files_to_save[10].write(str(test_pos_prec)+"\n")
            files_to_save[11].write(str(test_pos_recall)+"\n")
            files_to_save[12].write(str(test_neg_prec)+"\n")
            files_to_save[13].write(str(test_neg_recall)+"\n")

        return {"dom_pos_percent": test_dom_pos_percent, "sens_pos_percent": test_sens_pos_percent, "first_pred_acc": test_first_pred_acc, "dataset_acc": test_dataset_acc,
                "pos_prec": test_pos_prec, "pos_rec": test_pos_recall, "neg_prec": test_neg_prec, "neg_rec": test_neg_recall}

    def _print_dataset_stats(self):

        train_dom_pos_count = np.count_nonzero(self.y_train[self.train_dom_mask] == self.class_positive_value)
        print("Dominant class positive percentage for training set:", train_dom_pos_count/(self.train_dom_mask.sum()))

        train_sens_pos_count = np.count_nonzero(self.y_train[~self.train_dom_mask] == self.class_positive_value)
        print("Sensitive class positive percentage for training set:", train_sens_pos_count/((~self.train_dom_mask).sum()))

        test_dom_pos_count = np.count_nonzero(self.dataset_y_test[self.test_dom_mask] == self.class_positive_value)
        print("Dominant class positive percentage for testing set:", test_dom_pos_count/(self.test_dom_mask.sum()))

        test_sens_pos_count = np.count_nonzero(self.dataset_y_test[~self.test_dom_mask] == self.class_positive_value)
        print("Sensitive class positive percentage for testing set:", test_sens_pos_count/((~self.test_dom_mask).sum()))

        val_dom_pos_count = np.count_nonzero(self.dataset_y_val[self.val_dom_mask] == self.class_positive_value)
        print("Dominant class positive percentage for validation set:", val_dom_pos_count/(self.val_dom_mask.sum()))

        val_sens_pos_count = np.count_nonzero(self.dataset_y_val[~self.val_dom_mask] == self.class_positive_value)
        print("Sensitive class positive percentage for validation set:", val_sens_pos_count/((~self.val_dom_mask).sum()))



if __name__ == "__main__":

#---------------------------------------- parameters and dataframe loading ----------------------------------------#

    k_values_to_check = range(21, 22, 2)
    class_attr = "Income"
    class_negative_value = "Negative Class\n(<50K)"
    class_positive_value = "Positive Class\n(>=50K)"
    sensitive_attr = "Gender"
    dominant_class_value = "Dominant Attribute\n(Male)"
    sensitive_class_value = "Sensitive Attribute\n(Female)"
    exclude_sensitive_attr = True
    save_results_per_gap = False
    metrics_recalculate_gap = 1000000
    percentage_difs = [None] # np.arange(0.015, 0.0020, -0.0025)
    sensitive_increases = np.arange(0.05, 0.55, 0.05) # [0.25]
    check_random_removals_until_fair = True
    save_folder_name = "acs_income_gender"
    val_results_path = "results/" + save_folder_name + "/val_results/"
    test_results_path = "results/" + save_folder_name + "/test_results/"
    folds_num = 10
    val_split = 0.1 / (1 - (1/folds_num)) # to get validation set size equal to test set for the folds

    df = datasets.fetch_acs_income()["data"]
    df["Income"] = datasets.fetch_acs_income()["target"]
    df = df.sample(n=50000, random_state=42)
    df = df.reset_index(drop=True)
    df["Income"] = df["Income"].apply(lambda x: "Positive Class\n(>=50K)" if x>=50000 else "Negative Class\n(<50K)")
    df["Gender"] = df["SEX"].apply(lambda x: "Dominant Attribute\n(Male)" if x==1.0 else "Sensitive Attribute\n(Female)")
    df = df.drop(columns=["SEX"])

#------------------------------------------------------------------------------------------------------------------#

    print(df.describe(include="all"))
    save_dataset_pos_rates(df=df, class_name=class_attr, sensitive_feature_name=sensitive_attr, dominant_value=dominant_class_value, class_positive_value=class_positive_value, save_folder_path="results/"+save_folder_name)
    plot_dataset_stats(df=df, class_name=class_attr, sensitive_feature_name=sensitive_attr, save_folder="results/"+save_folder_name+"/plots/", name=save_folder_name)
    dominant_class_value, sensitive_class_value, class_negative_value, class_positive_value = encode_dataframe(df, sensitive_attr, dominant_class_value, sensitive_class_value, class_attr, class_negative_value, class_positive_value)
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=42)

    val_cross_val_results_path = val_results_path + str(folds_num) + "_fold_scv"
    test_cross_val_results_path = test_results_path + str(folds_num) + "_fold_scv"
    if not exclude_sensitive_attr:
        val_cross_val_results_path += "_with_sens_attr"
        test_cross_val_results_path += "_with_sens_attr"
    val_cross_val_results_path += "/"
    test_cross_val_results_path += "/"

    # save over k
    if len(k_values_to_check) > 1:
        if percentage_difs[0]:
            val_cross_val_results_path += "over_k/perc_dif_" + str(percentage_difs[0]) + "/"
            test_cross_val_results_path += "over_k/perc_dif_" + str(percentage_difs[0]) + "/"
        else:
            val_cross_val_results_path += "over_k/sens_incr_" + str(sensitive_increases[0]) + "/"
            test_cross_val_results_path += "over_k/sens_incr_" + str(sensitive_increases[0]) + "/"
        if check_random_removals_until_fair:
            val_cross_val_results_path += "random_removals_until_fair_10_fold_scv/"
            test_cross_val_results_path += "random_removals_until_fair_10_fold_scv/"
        if not os.path.exists(val_cross_val_results_path):
            os.makedirs(val_cross_val_results_path)  
        if not os.path.exists(test_cross_val_results_path):
            os.makedirs(test_cross_val_results_path)

        for k in k_values_to_check:
            print(f"\nRunning for k={k}")
            avg_val_dom_pos_percent = avg_val_sens_pos_percent = avg_val_first_pred_acc = avg_val_dataset_acc = avg_val_pos_prec = avg_val_pos_rec = avg_val_neg_prec = avg_val_neg_rec = avg_train_ids_removed = avg_percentage_of_train_ids_removed = 0
            avg_test_dom_pos_percent = avg_test_sens_pos_percent = avg_test_first_pred_acc = avg_test_dataset_acc = avg_test_pos_prec = avg_test_pos_rec = avg_test_neg_prec = avg_test_neg_rec = 0
            for fold, (train_val_idx, test_idx) in enumerate(skf.split(df.drop(class_attr, axis=1), df[class_attr]), 1):
                print(f"\nFold {fold}")
                x_train, x_val, y_train, dataset_y_val = train_test_split(df.drop(class_attr, axis=1).iloc[train_val_idx], 
                                                                        df[class_attr].iloc[train_val_idx], 
                                                                        test_size=val_split, 
                                                                        stratify=df[class_attr].iloc[train_val_idx],
                                                                        random_state=42) 
                x_test, dataset_y_test = df.drop(class_attr, axis=1).iloc[test_idx], df[class_attr].iloc[test_idx]
                
                print(f"\nTraining ids: {x_train.shape[0]}")
                print(f"Testing ids: {x_test.shape[0]}")
                print(f"Validation ids: {x_val.shape[0]}")

                knn_fair_model = FairKnnWithRemovals(k,
                                            class_attr,
                                            class_negative_value,
                                            class_positive_value,
                                            sensitive_attr,
                                            dominant_class_value,
                                            sensitive_class_value,
                                            exclude_sensitive_attr,
                                            save_results_per_gap,
                                            metrics_recalculate_gap,
                                            val_results_path,
                                            test_results_path,
                                            percentage_difs[0],
                                            sensitive_increases[0],
                                            check_random_removals_until_fair,
                                            x_train,
                                            x_val,
                                            x_test,
                                            y_train,
                                            dataset_y_val,
                                            dataset_y_test,
                                            val_cross_val_results_path,
                                            "fold_"+str(fold),
                                            str(k) + "_neigh_")
                fold_results = knn_fair_model.fit_predict_metrics()

                avg_val_dom_pos_percent += fold_results["validation"]["dom_pos_percent"]/folds_num
                avg_val_sens_pos_percent += fold_results["validation"]["sens_pos_percent"]/folds_num
                avg_val_first_pred_acc += fold_results["validation"]["first_pred_acc"]/folds_num
                avg_val_dataset_acc += fold_results["validation"]["dataset_acc"]/folds_num
                avg_val_pos_prec += fold_results["validation"]["pos_prec"]/folds_num
                avg_val_pos_rec += fold_results["validation"]["pos_rec"]/folds_num
                avg_val_neg_prec += fold_results["validation"]["neg_prec"]/folds_num
                avg_val_neg_rec += fold_results["validation"]["neg_rec"]/folds_num
                avg_train_ids_removed += fold_results["validation"]["train_ids_removed"]/folds_num
                avg_percentage_of_train_ids_removed += fold_results["validation"]["percentage_of_train_ids_removed"]/folds_num

                avg_test_dom_pos_percent += fold_results["test"]["dom_pos_percent"]/folds_num
                avg_test_sens_pos_percent += fold_results["test"]["sens_pos_percent"]/folds_num
                avg_test_first_pred_acc += fold_results["test"]["first_pred_acc"]/folds_num
                avg_test_dataset_acc += fold_results["test"]["dataset_acc"]/folds_num
                avg_test_pos_prec += fold_results["test"]["pos_prec"]/folds_num
                avg_test_pos_rec += fold_results["test"]["pos_rec"]/folds_num
                avg_test_neg_prec += fold_results["test"]["neg_prec"]/folds_num
                avg_test_neg_rec += fold_results["test"]["neg_rec"]/folds_num

            val_cross_val_results_file_name = val_cross_val_results_path + str(k) + "_neigh_" + str(folds_num) + "_folds_avg_val"
            test_cross_val_results__file_name = test_cross_val_results_path + str(k) + "_neigh_" + "_perc_dif_" + str(folds_num) + "_folds_avg_test"
            val_cross_val_results_file_name = val_cross_val_results_path + str(k) + "_neigh_" + str(folds_num) + "_folds_avg_val"
            test_cross_val_results__file_name = test_cross_val_results_path + str(k) + "_neigh_" + str(folds_num) + "_folds_avg_test"
            if not exclude_sensitive_attr:
                val_cross_val_results_file_name += "_with_sens_attr"
                test_cross_val_results__file_name += "_with_sens_attr"
            val_cross_val_results_file_name += ".txt"
            test_cross_val_results__file_name += ".txt"

            with open(val_cross_val_results_file_name, "w") as f1, \
                open(test_cross_val_results__file_name, "w") as f2:
            
                f1.write("avg_val_dom_pos_percent:" + str(avg_val_dom_pos_percent) + "\n")
                f1.write("avg_val_sens_pos_percent:" + str(avg_val_sens_pos_percent) + "\n")
                f1.write("avg_val_first_pred_acc:" + str(avg_val_first_pred_acc) + "\n")
                f1.write("avg_val_dataset_acc:" + str(avg_val_dataset_acc) + "\n")
                f1.write("avg_val_pos_prec:" + str(avg_val_pos_prec) + "\n")
                f1.write("avg_val_pos_rec:" + str(avg_val_pos_rec) + "\n")
                f1.write("avg_val_neg_prec:" + str(avg_val_neg_prec) + "\n")
                f1.write("avg_val_neg_rec:" + str(avg_val_neg_rec) + "\n")
                f1.write("avg_train_ids_removed:" + str(avg_train_ids_removed) + "\n")
                f1.write("avg_percentage_of_train_ids_removed:" + str(avg_percentage_of_train_ids_removed) + "\n")

                f2.write("avg_test_dom_pos_percent:" + str(avg_test_dom_pos_percent) + "\n")
                f2.write("avg_test_sens_pos_percent:" + str(avg_test_sens_pos_percent) + "\n")
                f2.write("avg_test_first_pred_acc:" + str(avg_test_first_pred_acc) + "\n")
                f2.write("avg_test_dataset_acc:" + str(avg_test_dataset_acc) + "\n")
                f2.write("avg_test_pos_prec:" + str(avg_test_pos_prec) + "\n")
                f2.write("avg_test_pos_rec:" + str(avg_test_pos_rec) + "\n")
                f2.write("avg_test_neg_prec:" + str(avg_test_neg_prec) + "\n")
                f2.write("avg_test_neg_rec:" + str(avg_test_neg_rec) + "\n")

            print(df.describe(include="all"))
    # save over sensitive increase or percentage dif
    else:
        values_to_iterate = []
        if percentage_difs[0]:
            val_cross_val_results_path += "over_perc_dif/" + str(k_values_to_check[0]) + "_neigh/"
            test_cross_val_results_path += "over_perc_dif/" + str(k_values_to_check[0]) + "_neigh/"
            values_to_iterate = [percentage_difs, [None]*len(percentage_difs)]
        else:
            val_cross_val_results_path += "over_sens_incr/" + str(k_values_to_check[0]) + "_neigh/"
            test_cross_val_results_path += "over_sens_incr/" + str(k_values_to_check[0]) + "_neigh/"
            values_to_iterate = [[None]*len(sensitive_increases), sensitive_increases]
        if check_random_removals_until_fair:
            val_cross_val_results_path += "random_removals_until_fair_10_fold_scv/"
            test_cross_val_results_path += "random_removals_until_fair_10_fold_scv/"
        if not os.path.exists(val_cross_val_results_path):
            os.makedirs(val_cross_val_results_path)  
        if not os.path.exists(test_cross_val_results_path):
            os.makedirs(test_cross_val_results_path)

        for percentage_dif, sensitive_increase in zip(*values_to_iterate):
            if percentage_dif:
                print(f"\nRunning for percentage_dif={percentage_dif} and k={k_values_to_check[0]}")
                removed_ids_extra_file_name = str(percentage_dif) + "_perc_dif_"
            else:
                print(f"\nRunning for sensitive_increase={sensitive_increase} and k={k_values_to_check[0]}")
                removed_ids_extra_file_name = str(sensitive_increase) + "_sens_incr_"
            avg_val_dom_pos_percent = avg_val_sens_pos_percent = avg_val_first_pred_acc = avg_val_dataset_acc = avg_val_pos_prec = avg_val_pos_rec = avg_val_neg_prec = avg_val_neg_rec = avg_train_ids_removed = avg_percentage_of_train_ids_removed= 0
            avg_test_dom_pos_percent = avg_test_sens_pos_percent = avg_test_first_pred_acc = avg_test_dataset_acc = avg_test_pos_prec = avg_test_pos_rec = avg_test_neg_prec = avg_test_neg_rec = 0
            for fold, (train_val_idx, test_idx) in enumerate(skf.split(df.drop(class_attr, axis=1), df[class_attr]), 1):
                print(f"\nFold {fold}")
                x_train, x_val, y_train, dataset_y_val = train_test_split(df.drop(class_attr, axis=1).iloc[train_val_idx], 
                                                                        df[class_attr].iloc[train_val_idx], 
                                                                        test_size=val_split, 
                                                                        stratify=df[class_attr].iloc[train_val_idx],
                                                                        random_state=42) 
                x_test, dataset_y_test = df.drop(class_attr, axis=1).iloc[test_idx], df[class_attr].iloc[test_idx]
                
                print(f"\nTraining ids: {x_train.shape[0]}")
                print(f"Testing ids: {x_test.shape[0]}")
                print(f"Validation ids: {x_val.shape[0]}")

                knn_fair_model = FairKnnWithRemovals(k_values_to_check[0],
                                            class_attr,
                                            class_negative_value,
                                            class_positive_value,
                                            sensitive_attr,
                                            dominant_class_value,
                                            sensitive_class_value,
                                            exclude_sensitive_attr,
                                            save_results_per_gap,
                                            metrics_recalculate_gap,
                                            val_results_path,
                                            test_results_path,
                                            percentage_dif,
                                            sensitive_increase,
                                            check_random_removals_until_fair,
                                            x_train,
                                            x_val,
                                            x_test,
                                            y_train,
                                            dataset_y_val,
                                            dataset_y_test,
                                            val_cross_val_results_path,
                                            "fold_"+str(fold),
                                            removed_ids_extra_file_name)
                fold_results = knn_fair_model.fit_predict_metrics()

                avg_val_dom_pos_percent += fold_results["validation"]["dom_pos_percent"]/folds_num
                avg_val_sens_pos_percent += fold_results["validation"]["sens_pos_percent"]/folds_num
                avg_val_first_pred_acc += fold_results["validation"]["first_pred_acc"]/folds_num
                avg_val_dataset_acc += fold_results["validation"]["dataset_acc"]/folds_num
                avg_val_pos_prec += fold_results["validation"]["pos_prec"]/folds_num
                avg_val_pos_rec += fold_results["validation"]["pos_rec"]/folds_num
                avg_val_neg_prec += fold_results["validation"]["neg_prec"]/folds_num
                avg_val_neg_rec += fold_results["validation"]["neg_rec"]/folds_num
                avg_train_ids_removed += fold_results["validation"]["train_ids_removed"]/folds_num
                avg_percentage_of_train_ids_removed += fold_results["validation"]["percentage_of_train_ids_removed"]/folds_num

                avg_test_dom_pos_percent += fold_results["test"]["dom_pos_percent"]/folds_num
                avg_test_sens_pos_percent += fold_results["test"]["sens_pos_percent"]/folds_num
                avg_test_first_pred_acc += fold_results["test"]["first_pred_acc"]/folds_num
                avg_test_dataset_acc += fold_results["test"]["dataset_acc"]/folds_num
                avg_test_pos_prec += fold_results["test"]["pos_prec"]/folds_num
                avg_test_pos_rec += fold_results["test"]["pos_rec"]/folds_num
                avg_test_neg_prec += fold_results["test"]["neg_prec"]/folds_num
                avg_test_neg_rec += fold_results["test"]["neg_rec"]/folds_num

            if percentage_dif:
                val_cross_val_results_file_name = val_cross_val_results_path + str(percentage_dif) + "_perc_dif_" + str(folds_num) + "_folds_avg_val"
                test_cross_val_results__file_name = test_cross_val_results_path + str(percentage_dif) + "_perc_dif_" + str(folds_num) + "_folds_avg_test"
            else:
                val_cross_val_results_file_name = val_cross_val_results_path + str(sensitive_increase) + "_sens_incr_" + str(folds_num) + "_folds_avg_val"
                test_cross_val_results__file_name = test_cross_val_results_path + str(sensitive_increase) + "_sens_incr_" + str(folds_num) + "_folds_avg_test"
            if not exclude_sensitive_attr:
                val_cross_val_results_file_name += "_with_sens_attr"
                test_cross_val_results__file_name += "_with_sens_attr"
            val_cross_val_results_file_name += ".txt"
            test_cross_val_results__file_name += ".txt"

            with open(val_cross_val_results_file_name, "w") as f1, \
                open(test_cross_val_results__file_name, "w") as f2:
            
                f1.write("avg_val_dom_pos_percent:" + str(avg_val_dom_pos_percent) + "\n")
                f1.write("avg_val_sens_pos_percent:" + str(avg_val_sens_pos_percent) + "\n")
                f1.write("avg_val_first_pred_acc:" + str(avg_val_first_pred_acc) + "\n")
                f1.write("avg_val_dataset_acc:" + str(avg_val_dataset_acc) + "\n")
                f1.write("avg_val_pos_prec:" + str(avg_val_pos_prec) + "\n")
                f1.write("avg_val_pos_rec:" + str(avg_val_pos_rec) + "\n")
                f1.write("avg_val_neg_prec:" + str(avg_val_neg_prec) + "\n")
                f1.write("avg_val_neg_rec:" + str(avg_val_neg_rec) + "\n")
                f1.write("avg_train_ids_removed:" + str(avg_train_ids_removed) + "\n")
                f1.write("avg_percentage_of_train_ids_removed:" + str(avg_percentage_of_train_ids_removed) + "\n")

                f2.write("avg_test_dom_pos_percent:" + str(avg_test_dom_pos_percent) + "\n")
                f2.write("avg_test_sens_pos_percent:" + str(avg_test_sens_pos_percent) + "\n")
                f2.write("avg_test_first_pred_acc:" + str(avg_test_first_pred_acc) + "\n")
                f2.write("avg_test_dataset_acc:" + str(avg_test_dataset_acc) + "\n")
                f2.write("avg_test_pos_prec:" + str(avg_test_pos_prec) + "\n")
                f2.write("avg_test_pos_rec:" + str(avg_test_pos_rec) + "\n")
                f2.write("avg_test_neg_prec:" + str(avg_test_neg_prec) + "\n")
                f2.write("avg_test_neg_rec:" + str(avg_test_neg_rec) + "\n")

            print(df.describe(include="all"))
