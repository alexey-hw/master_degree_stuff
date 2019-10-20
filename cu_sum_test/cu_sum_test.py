from math import log, inf, sqrt
from random import uniform, randint
from collections import namedtuple
from statistics import mean
from texttable import Texttable

AlternativeParameters = namedtuple('AlternativeParameters', 'a0 a1 h')
ConfidenceInterval = namedtuple('ConfidenceInterval', 'c0 c1')
SequentialTestResult = namedtuple('SequentialTestResult', 'observation_length accepted_hypothesis')
LabResult = namedtuple('LabResult', 'dist chosen_time time_diff alpha beta')
ALTERNATIVES = [AlternativeParameters((-1, 7), (2, 4), 15),
                AlternativeParameters((-1, 7), (0, 6), 10),
                AlternativeParameters((-1, 7), (-.9, 6.9), 5.4),
                AlternativeParameters((-1, 7), (-.99, 6.99), 1.68),
                AlternativeParameters((-1, 7), (-.999, 6.999), 0.24951)]
NUMBER_OF_SIMULATIONS = 1000
SAMPLE_LENGTH = 1000
TABLE_HEADER = ['distribution param', 'alternative param', 'Euclidean distance', 'mean test length',
                'mean time between chosen hyp. and real param. change', 'alpha', 'beta', 'h']


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


def run_seq_test(next_seq_val_supplier, density_function, alternative_params):
    used_observations = [next_seq_val_supplier(0)]
    log_like_sum = max(0., count_log_likelihood(used_observations[-1], density_function, alternative_params))
    log_likelihoods = [log_like_sum]
    min_log_like = log_like_sum
    num_of_iter = 1
    while (log_like_sum - min_log_like) < alternative_params.h and num_of_iter < SAMPLE_LENGTH:
        used_observations.append(next_seq_val_supplier(num_of_iter))
        log_likelihoods.append(count_log_likelihood(used_observations[-1], density_function, alternative_params))
        min_log_like = min(min_log_like, log_like_sum)
        log_like_sum += log_likelihoods[-1]
        log_like_sum = max(0., log_like_sum)
        num_of_iter += 1
    return SequentialTestResult(len(used_observations), int(num_of_iter < SAMPLE_LENGTH))


def create_val_supplier(jump_t, alternative_params):
    def val_supplier(cur_t):
        return uniform(*alternative_params.a0) if cur_t < jump_t else uniform(*alternative_params.a1)

    return val_supplier


def run_test_with_predefined_values(alternative_params):
    jump_t = randint(0, SAMPLE_LENGTH * 2)
    jump_t = jump_t if jump_t < SAMPLE_LENGTH else SAMPLE_LENGTH
    correct_hypothesis = int(jump_t < SAMPLE_LENGTH)
    return run_seq_test(create_val_supplier(jump_t, alternate_params),
                        lambda x, a, b: 1. / (b - a) if a <= x <= b else 0,
                        alternative_params), correct_hypothesis, jump_t


def run_simulations(alternate_params):
    simulation_results = [run_test_with_predefined_values(alternate_params) for _ in range(NUMBER_OF_SIMULATIONS)]
    param_dist = calculate_euclid_distance_for_params(alternate_params)
    alpha_count = sum(sim_res[1] == 0 and sim_res[0].accepted_hypothesis == 1 for sim_res in simulation_results)
    beta_count = sum(sim_res[1] == 1 and sim_res[0].accepted_hypothesis == 0 for sim_res in simulation_results)
    mean_chosen_time = mean(map(lambda simulation_result: simulation_result[0].observation_length, simulation_results))
    mean_time_diff = mean(map(lambda sim_res: abs(sim_res[0].observation_length - sim_res[2]), simulation_results))
    return LabResult(param_dist, mean_chosen_time, mean_time_diff, alpha_count / NUMBER_OF_SIMULATIONS,
                     beta_count / NUMBER_OF_SIMULATIONS)


def calculate_euclid_distance_for_params(alternate_params):
    return sqrt((alternate_params.a0[0] - alternate_params.a1[0]) ** 2 +
                (alternate_params.a0[1] - alternate_params.a1[1]) ** 2)


if __name__ == "__main__":
    table = Texttable()
    table.header(TABLE_HEADER)
    for alternate_params in ALTERNATIVES:
        sim_result = run_simulations(alternate_params)
        table.add_row(
            [alternate_params.a0, alternate_params.a1, sim_result.dist, sim_result.chosen_time, sim_result.time_diff,
             sim_result.alpha, sim_result.beta, alternate_params.h])
    print(table.draw())
