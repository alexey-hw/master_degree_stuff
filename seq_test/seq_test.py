from math import log, inf, sqrt
from random import uniform, randint
from collections import namedtuple
from statistics import mean
from texttable import Texttable


AlternativeParameters = namedtuple('AlternativeParameters', 'a0 a1')
ConfidenceInterval = namedtuple('ConfidenceInterval', 'c0 c1')
SequentialTestResult = namedtuple('SequentialTestResult', 'observation_length accepted_hypothesis')
LabResult = namedtuple('LabResult', 'dist conv_speed alpha beta')
ALTERNATIVES = [AlternativeParameters((-1, 7), (2, 4)),
                AlternativeParameters((-1, 7), (0, 6)),
                AlternativeParameters((-1, 7), (-.9, 6.9)),
                AlternativeParameters((-1, 7), (-.99, 6.99)),
                AlternativeParameters((-1, 7), (-.999, 6.999))]
ALPHA_0 = 0.05
BETA_0 = 0.05
NUMBER_OF_SIMULATIONS = 1000
TABLE_HEADER = ['distribution param', 'alternative param', 'Euclidean distance', 'mean convergence speed', 'alpha',
                     'beta']


def count_log_likelihood(observation, density_function, alternative_params):
    likelihood_with_param_0 = density_function(observation, *alternative_params.a0)
    if likelihood_with_param_0 == 0:
        return inf
    likelihood_with_param_1 = density_function(observation, *alternative_params.a1)
    if likelihood_with_param_1 == 0:
        return -inf
    return log(likelihood_with_param_1 / likelihood_with_param_0)


def count_confidence_interval(alpha, beta):
    return ConfidenceInterval(log(beta / (1. - alpha)), log((1. - beta) / alpha))


def run_seq_test(next_seq_val_supplier, density_function, alternative_params, confidence_interval):
    used_observations = [next_seq_val_supplier()]
    log_likelihood_sum = count_log_likelihood(used_observations[-1], density_function, alternative_params)
    log_likelihoods = [log_likelihood_sum]
    while confidence_interval.c1 >= log_likelihood_sum >= confidence_interval.c0:
        used_observations.append(next_seq_val_supplier())
        log_likelihoods.append(count_log_likelihood(used_observations[-1], density_function, alternative_params))
        log_likelihood_sum += log_likelihoods[-1]
    return SequentialTestResult(len(used_observations), int(confidence_interval.c1 < log_likelihood_sum))


def run_test_with_predefined_values(alternative_params):
    confidence_interval = count_confidence_interval(ALPHA_0, BETA_0)
    correct_hypothesis = randint(0, 1)
    shuffled_params = alternative_params if correct_hypothesis == 0 else AlternativeParameters(alternative_params.a1,
                                                                                               alternative_params.a0)
    return run_seq_test(lambda: uniform(0, 7),
                        lambda x, a, b: 1. / (b - a) if a <= x <= b else 0,
                        shuffled_params, confidence_interval), correct_hypothesis


def run_simulations(alternate_params):
    simulation_results = [run_test_with_predefined_values(alternate_params) for _ in range(NUMBER_OF_SIMULATIONS)]
    param_dist = calculate_euclid_distance_for_params(alternate_params)
    alpha_count = sum(sim_res[1] == 0 and sim_res[0].accepted_hypothesis == 1 for sim_res in simulation_results)
    beta_count = sum(sim_res[1] == 1 and sim_res[0].accepted_hypothesis == 0 for sim_res in simulation_results)
    mean_conv = mean(map(lambda simulation_result: simulation_result[0].observation_length, simulation_results))
    return LabResult(param_dist, mean_conv, alpha_count / NUMBER_OF_SIMULATIONS, beta_count / NUMBER_OF_SIMULATIONS)


def calculate_euclid_distance_for_params(alternate_params):
    return sqrt((alternate_params.a0[0] - alternate_params.a1[0]) ** 2 +
                (alternate_params.a0[1] - alternate_params.a1[1]) ** 2)


if __name__ == "__main__":
    table = Texttable()
    table.header(TABLE_HEADER)
    for alternate_params in ALTERNATIVES:
        sim_result = run_simulations(alternate_params)
        table.add_row(
            [alternate_params.a0, alternate_params.a1, sim_result.dist, sim_result.conv_speed, sim_result.alpha,
             sim_result.beta])
    print(table.draw())
