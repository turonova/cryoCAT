import numpy as np
import math


def compute_rmse(array1, array2):
    """Compute the Root Mean Square Error (RMSE) between two arrays along each column.

    Parameters
    ----------
    array1 : ndarray
        First input array.
    array2 : ndarray
        Second input array, must have the same shape as array1.

    Returns
    -------
    rmse : ndarray
        An array containing the RMSE computed for each column of the input arrays.

    Notes
    -----
    This function computes the RMSE by first calculating the squared differences between corresponding elements of the
    input arrays, then taking the mean of these squared differences along each column, and finally taking the square
    root of these means.

    Examples
    --------
    >>> array1 = np.array([[1, 2], [3, 4]])
    >>> array2 = np.array([[1, 1], [1, 1]])
    >>> compute_rmse(array1, array2)
    array([1.41421356, 2.12132034])
    """

    # Compute squared differences along each column
    squared_diff = (array1 - array2) ** 2

    # Calculate mean of squared differences along each column
    mean_squared_diff = np.mean(squared_diff, axis=0)

    # Take square root to obtain RMSE for each column
    rmse = np.sqrt(mean_squared_diff)

    return rmse


def get_all_pairs(input_numbers):
    """Generate all possible unique pairs from a list of numbers.

    Parameters
    ----------
    input_numbers : list
        List of integers or floats from which pairs are to be generated.

    Returns
    -------
    list
        A list of tuples, each containing a pair of numbers from the input list.

    Examples
    --------
    >>> get_all_pairs([1, 2, 3])
    [(1, 2), (1, 3), (2, 3)]
    """

    pairs = []
    for i in range(len(input_numbers)):
        for j in range(i + 1, len(input_numbers)):
            pairs.append((input_numbers[i], input_numbers[j]))
    return pairs


def otsu_threshold(input_values):
    """Calculate the Otsu threshold for binarization based on the histogram of input values.

    Parameters
    ----------
    input_values : ndarray
        An array of input values for which the histogram and threshold need to be computed.

    Returns
    -------
    float
        The computed threshold value according to Otsu's method.

    Notes
    -----
    Otsu's method is used to automatically perform histogram shape-based image thresholding.
    The algorithm assumes that the data contains two classes of values following a bimodal
    histogram, it then calculates the optimum threshold separating the two classes so that their combined spread
    (intra-class variance) is minimal.

    References
    ----------
    Taken from: https://www.kdnuggets.com/2018/10/basic-image-analysis-python-p4.html

    Examples
    --------
    >>> import numpy as np
    >>> input_values = np.random.randint(0, 256, 1000)
    >>> threshold = otsu_threshold(input_values)
    >>> print("Otsu's threshold:", threshold)
    """

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
    3
    """
    digit_length = len(str(input_number))

    if "." in str(input_number):
        digit_length -= 1

    return digit_length


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
