from dataframe import DataFrame
from methods.auxiliary_function import auxiliary_function, calculate_original_fit
from methods.fit_method import FitMethod
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import OLSInfluence
from methods.least_squares import OrdinaryLeastSquaresSM as my_ols


class WeightedLeastSquares(FitMethod):
    """Weighted least squares fit for approximation up to given power."""

    def __init__(self, power):
        self.power = power
        self.name = "Weighted least squares hybrid (pow=%d)" % self.power

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.source_matrix = np.column_stack(
            [data_frame.temp ** i if i != 0 else np.ones(len(data_frame.temp)) for i in range(0, self.power + 1)])

        (self.aux_values, self.aux_weights) = auxiliary_function(data_frame)
        self.aux_fit = sm.WLS(data_frame.experiment, self.source_matrix, weights=self.aux_weights).fit()

        self.fit_coefficients = self.aux_fit.params
        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack([i * data_frame.temp ** (i - 1) for i in range(0, self.power + 1)]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)

    def calculate_refpoints(self):
        t_ref_array = [1]
        t_ref_array.extend([self.data_frame.reference_temperature ** i for i in range(1, self.power + 1)])
        t_ref_diff_array = [self.data_frame.reference_temperature ** i for i in range(self.power)]
        diff_coefficients = self.fit_coefficients[1:]
        print(np.dot(t_ref_array, self.fit_coefficients), np.dot(t_ref_diff_array, diff_coefficients))


class WeightedLeastSquaresWithAuxiliaryFunction(FitMethod):
    """Weighted least squares fit for approximation up to given power using auxiliary function."""

    def __init__(self, power):
        self.power = power
        self.name = "WLS Aux, (pow=%d)" % self.power

    def fit(self, data_frame: DataFrame):
        self.temp = data_frame.dh_t
        self.experiment = data_frame.dh_e
        self.experiment_derivatives = data_frame.cp_e
        self.data_frame = data_frame

        (self.aux_values, self.aux_weights) = auxiliary_function(data_frame)
        self.aux_source_matrix = np.column_stack([self.temp ** i for i in range(1, self.power + 1)])
        self.aux_source_matrix = sm.add_constant(self.aux_source_matrix)

        self.aux_fit = sm.WLS(self.aux_values, self.aux_source_matrix, weights=self.aux_weights).fit()

        self.aux_fit_coefficients = self.aux_fit.params

        self.source_matrix = np.vstack(
            [self.temp ** i if i != 0 else np.ones(len(self.temp)) for i in range(-1, 5)]).T

        original_fit_dict = calculate_original_fit(
            {power: self.aux_fit_coefficients[power] for power in range(0, len(self.aux_fit_coefficients))},
            data_frame.reference_temperature,
            data_frame.reference_cvalue)
        self.original_fit_coefficients = [original_fit_dict[power] if power in original_fit_dict else 0.0 for power in
                                          range(-1, 5)]
        self.fit = np.dot(self.source_matrix, self.original_fit_coefficients)

        # self.derivative_matrix = np.vstack([i * self.temp ** (i - 1) for i in range(-1, 5)]).T
        self.derivative_matrix = np.vstack([i * data_frame.cp_t ** (i - 1) for i in range(-1, 5)]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.original_fit_coefficients)

    def calculate_derivative_residuals(self): # todo redo
        derivative_matrix = np.vstack([i * self.data_frame.cp_t ** (i - 1) for i in range(-1, 5)]).T
        derivative = np.dot(derivative_matrix, self.original_fit_coefficients)
        self.derivative_residuals = (self.data_frame.cp_e - derivative)/np.std(self.data_frame.cp_e - derivative)

    def plot_derivative_residuals(self, ax, **kwargs):
        ax.scatter(self.data_frame.cp_t, self.derivative_residuals, **kwargs)

    def calculate_refpoints(self):
        t_ref_array = np.array([self.data_frame.reference_temperature ** i
                                if i != 0 else 1 for i in range(-1, 5)])

        t_ref_diff_array = [-self.data_frame.reference_temperature ** -2]
        t_ref_diff_array.extend([i * self.data_frame.reference_temperature ** (i - 1) for i in range(5)])
        print(np.dot(t_ref_array, self.original_fit_coefficients),
              np.dot(t_ref_diff_array, self.original_fit_coefficients))

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        cp_temp = hc_data.temp
        cp_source_matrix = np.vstack([i * cp_temp ** (i - 1) for i in range(-1, 5)]).T
        fit_cp = np.dot(cp_source_matrix, self.original_fit_coefficients)
        return experiment_cp - fit_cp


class WeightedСonstrainedLeastSquaresSM(FitMethod):
    """experiments with weights"""

    def __init__(self, min_power, max_power):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"

        self.min_power = min_power
        self.max_power = max_power

        self.name = f"Weighted constrained least squares (powers {min_power} ... {max_power})"

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        t_ref = data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(data_frame.temp))

        c_0 = data_frame.reference_value
        c_1 = data_frame.reference_cvalue

        (self.aux_values, self.aux_weights) = auxiliary_function(data_frame)

        updated_experiment = data_frame.experiment - c_0 * np.ones(len(data_frame.temp)) - \
                             c_1 * (data_frame.temp - t_ref_vector)

        self.updated_matrix = np.column_stack(
            [data_frame.temp ** i + (i - 1) * t_ref_vector ** i - i * data_frame.temp * t_ref_vector ** (i - 1) \
             for i in range(self.min_power, self.max_power + 1) if i not in [0, 1]])

        ols_result = sm.OLS(updated_experiment, self.updated_matrix).fit()
        # cooks_distance_influential = 4/(len(self.data_frame.temp - (self.max_power - self.min_power) - 1))
        # ols_cooks_distance = OLSInfluence(ols_result).cooks_distance[1]
        ols_stud_residuals = OLSInfluence(ols_result).dfbetas
        # ols_influence = OLSInfluence(ols_result).influence

        w = np.ones(len(data_frame.temp))
        for residual, weight in zip(ols_stud_residuals, w):
            if residual > 2:
                w = 0.1
            else:
                w = 1

        self.aux_fit = sm.WLS(updated_experiment, self.updated_matrix, weights=w).fit()

        self.aux_coefficients = self.aux_fit.params

        a_1 = c_1 - \
              sum([i * self.aux_coefficients[i - self.min_power] * t_ref ** (i - 1) \
                   for i in range(self.min_power, 0)]) - \
              sum([i * self.aux_coefficients[i - 2 - self.min_power] * t_ref ** (i - 1) \
                   for i in range(2, self.max_power + 1)])
        a_0 = c_0 - a_1 * t_ref - \
              sum([self.aux_coefficients[i - self.min_power] * t_ref ** i \
                   for i in range(self.min_power, 0)]) - \
              sum([self.aux_coefficients[i - 2 - self.min_power] * t_ref ** i \
                   for i in range(2, self.max_power + 1)])

        self.fit_coefficients = []
        self.fit_coefficients.extend(self.aux_coefficients[:-self.min_power])
        self.fit_coefficients.extend([a_0, a_1])
        self.fit_coefficients.extend(self.aux_coefficients[-self.min_power:])

        self.source_matrix = np.vstack(
            [data_frame.temp ** i if i != 0 else np.ones(len(data_frame.temp)) for i in
             range(self.min_power, self.max_power + 1)]).T

        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack(
            [i * data_frame.temp ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)

    def calculate_refpoints(self):
        t_ref_array = [self.data_frame.reference_temperature ** i if i != 0
                       else 1 for i in range(self.min_power, self.max_power + 1)]

        if self.min_power < 0:
            t_ref_diff_array = [i * self.data_frame.reference_temperature ** (i - 1) for i in
                                range(self.min_power, 0)]
            t_ref_diff_array.extend(
                [i * self.data_frame.reference_temperature ** (i - 1) for i in range(self.max_power + 1)])
        else:
            t_ref_diff_array = [i * self.data_frame.reference_temperature ** (i - 1) for i in
                                range(self.min_power, self.max_power)]

        print(np.dot(t_ref_array, self.fit_coefficients),
              np.dot(t_ref_diff_array, self.fit_coefficients[:len(t_ref_diff_array)]))


class RobustCLS(FitMethod):
    """trying robust methods"""

    def __init__(self, min_power, max_power):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"

        self.min_power = min_power
        self.max_power = max_power

        self.name = f"Robust CLS (powers {min_power} ... {max_power})"

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        t_ref = data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(data_frame.temp))

        c_0 = data_frame.reference_value
        c_1 = data_frame.reference_cvalue

        updated_experiment = data_frame.experiment - c_0 * np.ones(len(data_frame.temp)) - \
                             c_1 * (data_frame.temp - t_ref_vector)

        self.updated_matrix = np.column_stack(
            [data_frame.temp ** i + (i - 1) * t_ref_vector ** i - i * data_frame.temp * t_ref_vector ** (i - 1) \
             for i in range(self.min_power, self.max_power + 1) if i not in [0, 1]])

        huber_t = sm.RLM(updated_experiment, self.updated_matrix, M=sm.robust.norms.HuberT())
        self.aux_fit = huber_t.fit()
        # self.aux_fit = huber_t.fit(cov="H2")
        #
        # other_norm = sm.RLM(updated_experiment, self.updated_matrix, M=sm.robust.norms.TrimmedMean())
        # self.aux_fit = other_norm.fit(scale_est=sm.robust.scale.HuberScale(), cov="H3")



        self.aux_coefficients = self.aux_fit.params

        a_1 = c_1 - \
              sum([i * self.aux_coefficients[i - self.min_power] * t_ref ** (i - 1) \
                   for i in range(self.min_power, 0)]) - \
              sum([i * self.aux_coefficients[i - 2 - self.min_power] * t_ref ** (i - 1) \
                   for i in range(2, self.max_power + 1)])
        a_0 = c_0 - a_1 * t_ref - \
              sum([self.aux_coefficients[i - self.min_power] * t_ref ** i \
                   for i in range(self.min_power, 0)]) - \
              sum([self.aux_coefficients[i - 2 - self.min_power] * t_ref ** i \
                   for i in range(2, self.max_power + 1)])

        self.fit_coefficients = []
        self.fit_coefficients.extend(self.aux_coefficients[:-self.min_power])
        self.fit_coefficients.extend([a_0, a_1])
        self.fit_coefficients.extend(self.aux_coefficients[-self.min_power:])

        self.source_matrix = np.vstack(
            [data_frame.temp ** i if i != 0 else np.ones(len(data_frame.temp)) for i in
             range(self.min_power, self.max_power + 1)]).T

        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack(
            [i * data_frame.temp ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)

    def calculate_refpoints(self):
        t_ref_array = [self.data_frame.reference_temperature ** i if i != 0
                       else 1 for i in range(self.min_power, self.max_power + 1)]

        if self.min_power < 0:
            t_ref_diff_array = [i * self.data_frame.reference_temperature ** (i - 1) for i in
                                range(self.min_power, 0)]
            t_ref_diff_array.extend(
                [i * self.data_frame.reference_temperature ** (i - 1) for i in range(self.max_power + 1)])
        else:
            t_ref_diff_array = [i * self.data_frame.reference_temperature ** (i - 1) for i in
                                range(self.min_power, self.max_power)]

        print(np.dot(t_ref_array, self.fit_coefficients),
              np.dot(t_ref_diff_array, self.fit_coefficients[:len(t_ref_diff_array)]))

