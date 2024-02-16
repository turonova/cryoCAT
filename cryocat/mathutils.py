import numpy as np


def get_all_pairs(input_numbers):
    pairs = []
    for i in range(len(input_numbers)):
        for j in range(i + 1, len(input_numbers)):
            pairs.append((input_numbers[i], input_numbers[j]))
    return pairs


def otsu_threshold(input_values):
    # Taken from: https://www.kdnuggets.com/2018/10/basic-image-analysis-python-p4.html

    stats_bins = np.histogram(input_values, bins=input_values.shape[0])
    bin_counts = stats_bins[0]
    s_max = (0, 0)

    for threshold in range(len(bin_counts)):
        # update
        w_0 = sum(bin_counts[:threshold])
        w_1 = sum(bin_counts[threshold:])

        mu_0 = sum([i * bin_counts[i] for i in range(0, threshold)]) / w_0 if w_0 > 0 else 0
        mu_1 = sum([i * bin_counts[i] for i in range(threshold, len(bin_counts))]) / w_1 if w_1 > 0 else 0

        # calculate - inter class variance
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2

        if s > s_max[1]:
            s_max = (threshold, s)

    return stats_bins[1][s_max[0]]


def get_number_of_digits(input_number):
    """Return the number of digits in the given input number.

    Parameters
    ----------
    input_number : int or float
        The number for which the number of digits needs to be calculated.

    Returns
    -------
    int
        The number of digits in the input number.

    Examples
    --------
    >>> get_number_of_digits(12345)
    5
    >>> get_number_of_digits(3.14)
    4
    """

    return len(str(input_number))
