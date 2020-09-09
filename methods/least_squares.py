from methods.fit_method import FitMethod
from dataframe import DataFrame
from methods.auxiliary_function import auxiliary_function, calculate_original_fit
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.gofplots import ProbPlot


class OrdinaryLeastSquaresSM(FitMethod):
    """Ordinary least squares fit for approximation up to given power. Basic method."""

    def __init__(self, power):
        self.power = power
        self.name = "Ordinary least squares (pow=%d)" % self.power

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.temp = data_frame.dh_t
        self.experiment = data_frame.dh_e

        self.source_matrix = np.column_stack(
            [data_frame.temp ** i if i != 0 else np.ones(len(self.temp)) for i in range(0, self.power + 1)])

        self.aux_fit = sm.OLS(self.experiment, self.source_matrix).fit()

        self.fit_coefficients = self.aux_fit.params
        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack([i * self.temp ** (i - 1) for i in range(0, self.power + 1)]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)

    def calculate_refpoints(self):
        t_ref_array = [1]
        t_ref_array.extend([self.data_frame.reference_temperature ** i for i in range(1, self.power + 1)])
        t_ref_diff_array = [self.data_frame.reference_temperature ** i for i in range(self.power)]
        diff_coefficients = self.fit_coefficients[1:]
        print(np.dot(t_ref_array, self.fit_coefficients), np.dot(t_ref_diff_array, diff_coefficients))

    def annotate_residuals(self, ax):
        qq = ProbPlot(self.residuals)
        sorted_residuals = np.flip(np.argsort(np.abs(self.residuals)), 0)
        top_3_residuals = sorted_residuals[:3]
        for r, i in enumerate(top_3_residuals):
            ax.annotate(self.temp[i],
                        xy=(np.sign(self.residuals[i]) * np.flip(qq.theoretical_quantiles, 0)[r], self.residuals[i]))

    def annotate_leverage(self, ax):
        leverage = self.aux_fit.get_influence().hat_matrix_diag
        cooks_distance = self.aux_fit.get_influence().cooks_distance[0]
        leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]

        for i in leverage_top_3:
            ax.annotate(self.data_frame.temp[i], xy=(leverage[i], self.residuals[i]))

    def annotate_cooks_distance(self, ax):
        influence = 4 / len(self.data_frame.temp)
        cooks_distance = self.aux_fit.get_influence().cooks_distance[0]
        leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]

        for i, distance in enumerate(cooks_distance):
            if distance > influence:
                ax.annotate(self.data_frame.temp[i], xy=(self.data_frame.temp[i], distance))

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        cp_temp = hc_data.temp
        cp_source_matrix = np.vstack([i * cp_temp ** (i - 1) for i in range(0, self.power + 1)]).T
        fit_cp = np.dot(cp_source_matrix, self.fit_coefficients)
        return experiment_cp - fit_cp


class СonstrainedLeastSquaresSM(FitMethod):
    """Constrained least squares fit for approximation up to given power."""

    def __init__(self, min_power, max_power):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"

        self.min_power = min_power
        self.max_power = max_power

        self.name = f"Constrained LS (powers {min_power} ... {max_power})"

    def fit(self, data_frame: DataFrame):

        self.data_frame = data_frame

        self.temp = data_frame.dh_t
        self.experiment = data_frame.dh_e
        self.cp_temp = data_frame.cp_t

        t_ref = data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.temp))

        c_0 = data_frame.reference_value
        c_1 = data_frame.reference_cvalue

        updated_experiment = self.experiment - c_0 * np.ones(len(self.temp)) - \
                             c_1 * (self.temp - t_ref_vector)

        self.updated_matrix = np.column_stack(
            [self.temp ** i + (i - 1) * t_ref_vector ** i - i * self.temp * t_ref_vector ** (i - 1) \
             for i in range(self.min_power, self.max_power + 1) if i not in [0, 1]])

        self.aux_fit = sm.OLS(updated_experiment, self.updated_matrix).fit()
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
            [self.temp ** i if i != 0 else np.ones(len(self.temp)) for i in
             range(self.min_power, self.max_power + 1)]).T

        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack(
            [i * self.cp_temp ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        cp_temp = hc_data.temp
        cp_source_matrix = np.vstack(
            [i * cp_temp ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        fit_cp = np.dot(cp_source_matrix, self.fit_coefficients)
        return experiment_cp - fit_cp

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

    def calculate_derivative_residuals(self):
        derivative_matrix = np.vstack(
            [i * self.cp_temp ** (i - 1) for i in range(self.min_power, self.max_power + 1)]).T
        derivative = np.dot(derivative_matrix, self.fit_coefficients)
        self.derivative_residuals = (self.data_frame.cp_e - derivative) / np.std(self.data_frame.cp_e - derivative)


class LeastSquaresFitNoFree(FitMethod):
    """Least squares fit for no-free-term polynomial."""

    def __init__(self):
        self.name = "No free term"

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.source_matrix = np.vstack(
            [data_frame.temp ** i - data_frame.reference_temperature ** i for i in [-1, 1, 2]]).T

        self.aux_fit = sm.OLS(data_frame.experiment, self.source_matrix).fit()
        self.fit_coefficients = self.aux_fit.params

        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack([i * data_frame.temp ** (i - 1) for i in [-1, 1, 2]]).T
        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)

    def calculate_refpoints(self):
        # t_ref_array = [1]
        # t_ref_array.extend(np.array([self.data_frame.reference_temperature ** i for i in range(1, self.power + 1)]))
        t_ref_diff_array = [(i + 1) * self.data_frame.reference_temperature ** i for i in [-2, 0, 1]]
        self.fit_coefficients
        print(0, np.dot(t_ref_diff_array, self.fit_coefficients))


class OrdinaryLeastSquaresSMminOne(FitMethod):
    """Ordinary least squares fit for approximation up to given power. Basic method."""

    def __init__(self, power):
        self.power = power
        self.name = "OLS -1 & sqrt (pow=%d)" % self.power

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.source_matrix = np.column_stack(
            [data_frame.temp ** i if i != 0 else np.ones(len(data_frame.temp)) for i in range(-1, self.power + 1)])

        self.source_matrix = np.column_stack((self.source_matrix, np.sqrt(data_frame.temp)))

        self.aux_fit = sm.OLS(data_frame.experiment, self.source_matrix).fit()

        self.fit_coefficients = self.aux_fit.params
        self.fit = np.dot(self.source_matrix, self.fit_coefficients)

        self.derivative_matrix = np.vstack([i * data_frame.temp ** (i - 1) for i in (range(-1, self.power + 1))])
        self.derivative_matrix = np.vstack((self.derivative_matrix, [(2*np.sqrt(data_frame.temp))**(-1)])).T

        self.fit_derivative = np.dot(self.derivative_matrix, self.fit_coefficients)

    def calculate_refpoints(self):
        t_ref_array = [1]
        t_ref_array.extend([self.data_frame.reference_temperature ** i for i in range(1, self.power + 1)])
        t_ref_diff_array = [self.data_frame.reference_temperature ** i for i in range(self.power)]
        diff_coefficients = self.fit_coefficients[1:]
        print(np.dot(t_ref_array, self.fit_coefficients), np.dot(t_ref_diff_array, diff_coefficients))



