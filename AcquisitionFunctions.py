import scipy
import numpy


def probability_of_improvement(mean, std, f_max, epsilon=0):
    Z = (mean - f_max - epsilon) / std
    output = scipy.stats.norm.cdf(Z)
    return output


def expected_improvement(mean, std, f_max, epsilon=0):
    Z = (mean - f_max - epsilon) / std
    return (mean - f_max - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)


def upper_confidence_bound(mean, std, t, d=1, v=1, delta=.1):
    amplifier = numpy.sqrt(v * (2 * numpy.log((t ** (d / 2. + 2)) * (numpy.pi ** 2) / (3. * delta))))
    print("Amplifier: ", amplifier)

    return mean + amplifier * std


def upper_confidence_bound_with_vanishing_amp(mean, std, num_of_iter):
    amplifier = 4 * (1 - (num_of_iter - 3) / 10)
    print("Amplifier: ", amplifier)

    return mean + amplifier * std
