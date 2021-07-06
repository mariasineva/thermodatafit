from dataframe import DataFrame
import numpy as np


def auxiliary_function(data_frame: DataFrame):
    """Calculates auxiliary function.

    Args:
        data_frame (DataFrame): data frame with experimental data

    Returns:
        (np.array, np.array): first array is auxiliary function values and seconds is their weights."""
    c_zero = data_frame.reference_heat_capacity_value
    is_weight_calculated = data_frame.experiment_weight > 0 or data_frame.reference_heat_capacity_error > 0

    aux_values = []
    aux_weights = []

    for (t_exp, dh_exp) in zip(data_frame.dh_t, data_frame.dh_e):
        dt = t_exp - data_frame.reference_temperature

        aux_values.append((t_exp / dt ** 2) * (dh_exp - dt * c_zero))

        if is_weight_calculated:
            aux_weights.append(((t_exp / dt ** 2) * (
                    dh_exp * data_frame.experiment_weight + dt * data_frame.reference_heat_capacity_error)) ** -2)
        else:
            aux_weights.append(10 ** -5)

    return np.array(aux_values), np.array(aux_weights)


def calculate_original_fit(aux_fit, t_ref, c_zero):
    """Calculates fit for original function given fit for auxiliary function.

    Args:
        aux_fit: dictionary { power: aux fit coefficient }
        t_ref (double): reference temperature point
        c_zero (double): original function derivative value at t_ref

    Returns:
        dictionary { power: original fit coefficient }."""
    # aux:       0 1 2 3 4
    #            A B C D E
    # result: -1 0 1 2 3 4 5
    #          c d a b e f g

    # g = E                             = 0;
    # f = D - 2 * g * t                 = D;
    # e = C - 2 * f * t - 3 * g * t * t = C - 2 * D * t;
    # c = A * t ** 2                    = A * t ** 2;
    # b = B - 2 * e * t - 3 * f * t ** 2 - 4 * g * t ** 3
    #   = B - 2 * C * t + D * t ** 2;
    # a = c_zero - 2 * b * t + A - 3 * e * t ** 2 - 4 * f * t ** 3 - 5 * g * t ** 4
    #   = c_zero + A - 2 * b * t - 3 * e * t ** 2 - 4 * D * t ** 3;
    # d = -c_zero * t - 2 * A * t + b * t ** 2 + 2 * e * t ** 3 + 3 * f * t ** 4 + 4 * g * t ** 5
    #   = -(2 * A + c_zero) * t + b * t ** 2 + 2 * e * t ** 3 + 3 * D * t ** 4;

    (A, B, C, D, E) = [aux_fit[power] if power in aux_fit else 0.0 for power in range(0, 5)]
    result_fit = {power: 0.0 for power in range(-1, 5)}

    result_fit[-1] = A * t_ref ** 2
    b = result_fit[2] = B - 2 * C * t_ref + D * t_ref ** 2;
    e = result_fit[3] = C - 2 * D * t_ref
    result_fit[4] = D

    result_fit[0] = -(2 * A + c_zero) * t_ref + b * t_ref ** 2 + 2 * e * t_ref ** 3 + 3 * D * t_ref ** 4
    result_fit[1] = c_zero + A - 2 * b * t_ref - 3 * e * t_ref ** 2 - 4 * D * t_ref ** 3

    return result_fit


def cost_function_least_abs_dev(params, matrix, experiment_data):
    return np.sum(np.abs(experiment_data - matrix.dot(params)))
