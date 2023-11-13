def otsu_threshold(bin_counts):
    # Taken from: https://www.kdnuggets.com/2018/10/basic-image-analysis-python-p4.html
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

    return s_max[0]


def get_number_of_digits(input_number):
    return len(str(input_number))
