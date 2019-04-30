import numpy as np


def false_positive_rate_at_95_recall(true_labels, distances):
    # type: (np.ndarray, np.ndarray)->float
    """
    Returns the false positive rate at 95% recall given the labels for each test pair and the computed distance
    between the images in each test pair.
    :param true_labels: 1xN ndarray of labels for each test pair
    :param distances: 1xN ndarray of distances for each test pair
    :return: the false positive rate for the test set
    """
    recall_point = 0.95  # ratio of matching pairs to recover out of the total number of matching pairs
    # sort labels on computed distances (smaller distance indicates likely match)
    # if the classifier was perfect the array would consist of all 1's followed by all 0's
#    import psutil
#    print(psutil.virtual_memory())
    sort_idx = np.argsort(distances)
#    print(psutil.virtual_memory())
    true_labels2 = true_labels[sort_idx]
    
    os.exit(1)

    # The number of matching pairs required to achieve a recall rate equal to recall_point
    num_req_true_matches = recall_point * np.sum(true_labels)

    # index into true_labels at which the required recall is obtained
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    # argmax is used as there doesn't exist a 'return on first occurence' find/where implementation
    # https://github.com/numpy/numpy/issues/2269
    threshold_index = np.argmax(np.cumsum(true_labels) >= num_req_true_matches)

    # number of negatives that would fall below the distance threshold and be considered matches
    false_positives = np.sum(true_labels[:threshold_index] == 0)
    # number of negatives that would lie above the distance threshold and be considered non-matches
    true_negatives = np.sum(true_labels[threshold_index:] == 0)

    # ratio of false matches to all non-matches
    false_positive_rate = float(false_positives) / float(false_positives + true_negatives)

    return false_positive_rate
