import sys
from collections import Counter

import numpy as np


def generate_cross_validation_partition(group_label, n_folds=5, n_repeats=1, portions=None, random_seed=None):
    '''
    This function generates partition indices of samples for cross-validation analysis.

    Parameters:
    -----------
    group_label: 1-D array or list of group labels of samples. If there are no groups in samples, a list of
        sample indices can be supplied for generating partitions based on individual samples rather than sample groups.
    n_folds: positive integer larger than 1, indicating the number of folds for cross-validation. Default is 5.
    n_repeats: positive integer, indicating how many times the n_folds cross-validation should be repeated.
        So the total number of cross-validation trials is n_folds * n_repeats. Default is 1.
    portions: 1-D array or list of positive integers, indicating the number of data folds in each set
        (e.g. training set, testing set, or validation set) after partitioning. The summation of elements
        in portions must be equal to n_folds. Default is [1, n_folds - 1].
    random_seed: positive integer, the seed for random generator. Default is None.

    Returns:
    --------
    partition: list of n_folds * n_repeats lists, each of which contains len(portions) sample index lists for
        a cross-validation trial.
    '''

    group_counter = Counter(group_label)
    unique_label = np.array(list(group_counter.keys()))
    n_group = len(unique_label)
    if n_group < n_folds:
        print('The number of groups in labels can not be smaller than the number of folds.')
        sys.exit(1)
    sorted_label = np.array(sorted(unique_label, key=lambda x: group_counter[x], reverse=True))

    if portions is None:
        portions = [1, n_folds - 1]
    else:
        if np.sum(portions) != n_folds:
            print('The summation of elements in portions must be equal to n_folds')
            sys.exit(1)

    if random_seed is not None:
        np.random.seed(random_seed)

    n_set = len(portions)
    partition = []
    for r in range(n_repeats):

        if r == 0 and random_seed is None:
            label = sorted_label.copy()
        else:
            idr = np.random.permutation(n_group)
            label = sorted_label[idr]

        folds = [[] for _ in range(n_folds)]
        fold_size = np.zeros((n_folds, ))

        for g in range(n_group):
            f = np.argmin(fold_size)
            folds[f].append(label[g])
            fold_size[f] += group_counter[label[g]]

        for f in range(n_folds):
            folds[f] = list(np.where(np.isin(group_label, folds[f]))[0])

        a = list(range(n_folds)) + list(range(n_folds))
        for f in range(n_folds):
            temp = []
            end = f
            for s in range(n_set):
                start = end
                end = start + portions[s]
                t = []
                for i in range(start, end):
                    t = t + folds[a[i]]
                temp.append(sorted(t))
            partition.append(temp)

    return partition



def replace_ccl_name(data, keep_id, ccl_info):
    o_id = np.sort(np.setdiff1d(range(data.shape[0]), keep_id))
    new_sample_name = []
    k_id = []
    for i in o_id:
        idi = np.where(ccl_info.other_id == data.iloc[i, 0])[0]
        if len(idi) > 0:
            new_sample_name.append(ccl_info.iloc[idi[0], :].improve_sample_id)
            k_id.append(i)
    # a = [[new_sample_name[i], data.iloc[k_id[i], 0]] for i in range(len(new_sample_name))]
    # print(a)
    data.iloc[k_id, 0] = new_sample_name

    data = data.iloc[keep_id + k_id, :]

    return data


# def replace_ccl_name(data, keep_id, ccl_info):
#     o_id = np.sort(np.setdiff1d(range(data.shape[0]), keep_id))
#     for i in o_id:
#         idi = np.where(ccl_info.other_id == data.iloc[i, 0])[0]
#         if len(idi) > 0:
#             data.iloc[i, 0] = ccl_info.iloc[idi[0], :].improve_sample_id
#             keep_id.append(i)
#     data = data.iloc[keep_id, :]
#
#     return data






# def replace_ccl_name(data, keep_id, ccl_info):
#     other_id = np.sort(np.setdiff1d(range(data.shape[0]), keep_id))
#     for i in other_id:
#         idi = np.where(ccl_info.other_id == data.iloc[i, 0])[0]
#         if len(idi) > 0:
#             data.iloc[i, 0] = ccl_info.iloc[idi[0], :].improve_sample_id
#             keep_id.append(i)
#     data = data.iloc[keep_id, :]
#
#     return data