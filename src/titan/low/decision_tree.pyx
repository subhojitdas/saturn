cimport cython
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple split_indices(np.ndarray[np.float64_t, ndim=1] feature_column, double threshold):
    """ Return indices of left and right split instead of full array copies """
    cdef int i, n = feature_column.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] left_indices = np.empty(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] right_indices = np.empty(n, dtype=np.int32)

    cdef int left_count = 0
    cdef int right_count = 0

    for i in range(n):
        if feature_column[i] <= threshold:
            left_indices[left_count] = i
            left_count += 1
        else:
            right_indices[right_count] = i
            right_count += 1

    return left_indices[:left_count], right_indices[:right_count]  # Trim unused space

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double information_gain(np.ndarray[np.float64_t, ndim=1] parent,
                              np.ndarray[np.float64_t, ndim=1] l_child,
                              np.ndarray[np.float64_t, ndim=1] r_child):
    print("information gain started")
    cdef double weight_l = len(l_child) / len(parent)
    cdef double weight_r = len(r_child) / len(parent)
    cdef double d = gini_index(parent) - (weight_l * gini_index(l_child) + weight_r * gini_index(r_child))
    print("information gain finished")
    return d

cpdef double gini_index(np.ndarray[np.float64_t, ndim=1] y):
    cdef dict label_counts = {}
    cdef int i, n = y.shape[0]
    cdef double p, gini = 1.0

    for i in range(n):
        label_counts[y[i]] = label_counts.get(y[i], 0) + 1  # Count occurrences

    for count in label_counts.values():
        p = count / n
        gini -= p * p  # Compute Gini index

    return gini




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple find_best_split(np.ndarray[np.float64_t, ndim=2] dataset, int feature_count):
    """ Optimized function to find the best split """

    cdef int sample_count = dataset.shape[0]
    cdef int feature_index
    cdef double threshold
    cdef double curr_info_gain, max_info_gain = -1.0  # Track best gain
    cdef np.ndarray[np.float64_t, ndim=1] feature_values
    cdef np.ndarray[np.int32_t, ndim=1] left_indices, right_indices
    cdef int left_size, right_size

    print(f"find_best_split started. sample_count: {sample_count}")

    cdef int best_feature_index = -1
    cdef double best_threshold = 0.0

    for feature_index in range(feature_count):
        feature_values = dataset[:, feature_index]  # Extract column
        possible_thresholds = np.sort(feature_values)[np.concatenate(([True], feature_values[1:] != feature_values[:-1]))]  # Faster unique

        for threshold in possible_thresholds:
            left_indices, right_indices = split_indices(dataset[:, feature_index], threshold)

            left_size = left_indices.shape[0]
            right_size = right_indices.shape[0]

            if left_size > 0 and right_size > 0:
                curr_info_gain = information_gain(dataset[:, -1], dataset[left_indices, -1], dataset[right_indices, -1])

                if curr_info_gain > max_info_gain:
                    max_info_gain = curr_info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

    print(f"find_best_split finished.")
    return best_feature_index, best_threshold, max_info_gain



