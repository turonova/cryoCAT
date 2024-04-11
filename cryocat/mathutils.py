import numpy as np
import math


def compute_rmse(array1, array2):
    # Compute squared differences along each column
    squared_diff = (array1 - array2) ** 2

    # Calculate mean of squared differences along each column
    mean_squared_diff = np.mean(squared_diff, axis=0)

    # Take square root to obtain RMSE for each column
    rmse = np.sqrt(mean_squared_diff)

    return rmse


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


def get_similar_size_factors(number, order="ascending"):
    """Return two factors of a number that are closest in size - either in ascending or descending order.

    Parameters
    ----------
    number: int
        The number for which to find factors.
    order: str, default="ascending"
        Order in which the numbers should be returned.

    Returns
    -------
    tuple:
        A tuple containing two factors of the number that are closest in size sorted by specified order.
        If no factors are found, returns the number itself and 1 (also sorted based on the specified order).
    """

    def sort(a, b):
        if order == "ascending":
            return min(a, b), max(a, b)
        else:
            return max(a, b), min(a, b)

    sqrt_num = int(math.sqrt(number))
    for i in range(sqrt_num, 1, -1):
        if number % i == 0:
            # If the number is divisible by i, return i and number // i
            return sort(i, number // i)
    # If no factors are found, return the number itself and 1
    return sort(number, 1)
