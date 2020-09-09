from dataframe import DataFrame
from methods.fit_method import FitMethod
import numpy as np
import scipy as sp
from scipy.optimize import least_squares as scipy_ls
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import t as students_t


class EinsteinPlankMethod(FitMethod):
    """list of functions"""

    def __init__(self):
        self.CONST_R = 8.3144598
        self.initial_params = [0.01, 1.00]
        self.initial_boundaries = [[0.0, 0.0], [500.0, 1.0e5]]
        self.initial_temperature = 298.15
        self.params = self.initial_params
        self.bounds = self.initial_boundaries

    def cp_cost(self, parameters, temperature, experiment):
        cp_calculated = 0.0
        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            ex = np.exp(x)
            cp_calculated += 3 * alpha * ex * x * x / (ex - 1) ** 2

        # print(self.name, "\te_cp:\t", sum(
        #     ((self.cp_draw(parameters, temperature) - experiment) / experiment) ** 2))
        return (cp_calculated * self.CONST_R - experiment) / experiment
        # return cp_calculated * self.CONST_R - experiment

    def h_cost(self, parameters, temperature, experiment):
        h_calculated = 0.0

        if len(parameters) % 2 != 0:
            print("Error!\n")

        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            h_calculated += 3 * alpha * x / (np.exp(x) - 1) * temperature

        h_calculated *= self.CONST_R
        return h_calculated - experiment

    def cp_draw(self, parameters, temperature):
        cp_calculated = 0.0
        for alpha, theta in zip(parameters[::2], parameters[1::2]):
            x = theta / temperature
            ex = np.exp(x)
            cp_calculated += 3 * alpha * ex * x * x / (ex - 1) ** 2
        return cp_calculated * self.CONST_R

    def h_draw(self, parameters, temperature):
        h_calculated = 0.
        if len(parameters) % 2 != 0:
            print("Error!\n")
        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            h_calculated += 3 * alpha * x / (np.exp(x) - 1) * temperature

        return h_calculated * self.CONST_R

    def dh_draw(self, parameters, temperature):
        return self.h_draw(parameters, temperature) - self.h_draw(parameters, self.initial_temperature)

    def dh_cost(self, parameters, temperature, experiment):
        # print(self.name, "\te_dh:\t", sum(
        #     ((self.h_draw(parameters, temperature) - self.h_draw(parameters, 298.15) - experiment) / experiment) ** 2))
        return (self.h_draw(parameters, temperature) - self.h_draw(parameters, 298.15) - experiment) / experiment

    def cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.dh_cost(parameters, h_temp, h_experiment), self.cp_cost(parameters, cp_temp, cp_experiment)
        return np.concatenate((dh, cp), axis=0)


class EinsteinPlankSum(EinsteinPlankMethod):
    """From Voronin."""

    def __init__(self, power, mode='h'):
        self.power = power
        self.name = "E-P.: %d terms, %s mode" % (self.power, mode)
        self.mode = mode
        # todo read about inheritance
        self.CONST_R = 8.3144598
        self.initial_params = [0.01, 1.00]
        self.initial_boundaries = [[0.0, 0.0], [500.0, 1.0e5]]
        self.initial_temperature = 298.15
        self.params = self.initial_params
        self.bounds = self.initial_boundaries

        self.confidence_params = [0.0, 0.0]

    def level_t(self):
        return students_t.isf(0.975, len(self.data_frame.temp) - len(self.params) / 2)

    def approx(self):
        for i in range(0, self.power - 1):
            self.params.append(0.01)
            self.params.append(1.00)
            self.bounds[0].append(0.0)
            self.bounds[0].append(0.0)
            self.bounds[1].append(500.0)
            self.bounds[1].append(1.0e5)

            if self.mode == 'j':
                res_lsq = scipy_ls(self.cost_function, self.params, bounds=self.bounds,
                                   args=(self.data_frame.dh_t, self.data_frame.dh_e,
                                         self.data_frame.cp_t, self.data_frame.cp_e))
            elif self.mode == 'h':
                res_lsq = scipy_ls(self.dh_cost, self.params, bounds=self.bounds,
                                   args=(self.data_frame.dh_t, self.data_frame.dh_e))
            elif self.mode == 'c':
                res_lsq = scipy_ls(self.cp_cost, self.params, bounds=self.bounds,
                                   args=(self.data_frame.cp_t, self.data_frame.cp_e))
            else:
                print('Wrong calculation type!')
                raise ValueError('Type can only be h c j.\n')

            # print(res_lsq.x.tolist())

        self.params = res_lsq.x.tolist()

    def covariance_stuff(self, cost_function):
        ls_zero_step = scipy_ls(cost_function, self.params, args=(self.data_frame.temp, self.data_frame.experiment))
        temporary_parameters = ls_zero_step.x
        jacobian = ls_zero_step.jac

        covariance_matrix = np.linalg.inv(jacobian.T.dot(jacobian))

        temporary_fit = cost_function(temporary_parameters, self.data_frame.temp, self.data_frame.experiment)
        temporary_fit *= temporary_fit
        # todo read https://stackoverflow.com/questions/28702631/scipy-curve-fit-returns-negative-variance
        sigma_squared = sum(temporary_fit) / (len(self.data_frame.temp) - len(self.params) / 2)
        covariance_matrix *= sigma_squared

        diagonal = np.sqrt(np.diagonal(covariance_matrix))
        self.confidence_params = np.sqrt(diagonal)
        self.confidence_params *= np.abs(self.level_t())
        print('Confidence something: ', self.confidence_params)

        need_new = False
        for a, b in zip(self.params, self.confidence_params):
            if a < b:
                need_new = True
        if need_new:
            self.params = np.append(self.params, 0.1)
            self.params = np.append(self.params, 10)

    def fit(self, data_frame: DataFrame):

        self.data_frame = data_frame

        self.approx()

        self.temp = data_frame.dh_t

        self.cp_temp = data_frame.cp_t

        self.fit = self.dh_draw(self.params, self.data_frame.dh_t)

        if not len(self.data_frame.cp_t):
            self.fit_derivative = self.cp_draw(self.params, self.data_frame.dh_t)
        else:
            self.fit_derivative = self.cp_draw(self.params, self.cp_temp)

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        cp_temp = hc_data.temp
        fit_cp = self.cp_draw(self.params, cp_temp)
        return experiment_cp - fit_cp

    def calculate_residuals(self):
        # if self.mode == 'h':
        self.residuals = (self.data_frame.dh_e - self.fit) / np.std(self.data_frame.dh_e - self.fit)
        # elif self.mode == 'c':
        #     self.residuals = (self.data_frame.cp_e - self.fit_derivative) / \
        #                      np.std(self.data_frame.cp_e - self.fit_derivative)
        # elif self.mode == 'j':
        #    self.residuals = np.concatenate((
        #        (self.data_frame.dh_e - self.fit) / np.std(self.data_frame.dh_e - self.fit),
        #        (self.data_frame.cp_e - self.fit_derivative) / np.std(self.data_frame.cp_e - self.fit_derivative)),
        #        axis=0)

    def calculate_derivative_residuals(self):
        self.derivative_residuals = (self.data_frame.cp_e - self.fit_derivative) / \
                                    np.std(self.data_frame.cp_e - self.fit_derivative)

    def plot_residuals(self, ax, **kwargs):
        """Plot standartised residuals using matplotlib."""
        # if self.mode == 'h':
        ax.scatter(self.data_frame.dh_t, self.residuals, **kwargs)
        # elif self.mode == 'c':
        #     ax.scatter(self.data_frame.cp_t, self.residuals, **kwargs)
        # elif self.mode == 'j':
        #     ax.scatter(np.concatenate((self.data_frame.dh_t, self.data_frame.cp_t), axis=0), self.residuals, **kwargs)

    # def plot_derivative(self, ax, **kwargs):
    #     """Plot derivative of the fit result using matplotlib."""
    #     ax.plot(self.temp, self.cp_draw(self.params, self.temp), **kwargs)


class EinsteinPlankAndPolynom(FitMethod):
    """list of functions"""

    def __init__(self):
        self.CONST_R = 8.3144598
        self.initial_params = [0.01, 1.00]
        self.initial_boundaries = [[0.0, 0.0], [500.0, 1.0e5]]
        self.initial_temperature = 298.15
        self.params = self.initial_params
        self.bounds = self.initial_boundaries

    def cp_cost(self, parameters, temperature, experiment):
        cp_calculated = 0.0
        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            ex = np.exp(x)
            cp_calculated += 3 * alpha * ex * x * x / (ex - 1) ** 2
        # return (cp_calculated * CONST_R - experiment)/experiment
        return cp_calculated * self.CONST_R - experiment

    def h_cost(self, parameters, temperature, experiment):
        h_calculated = 0.0

        if len(parameters) % 2 != 0:
            print("Error!\n")

        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            h_calculated += 3 * alpha * x / (np.exp(x) - 1) * temperature

        h_calculated *= self.CONST_R
        return h_calculated - experiment

    def cp_draw(self, parameters, temperature):
        cp_calculated = 0.0
        for alpha, theta in zip(parameters[::2], parameters[1::2]):
            x = theta / temperature
            ex = np.exp(x)
            cp_calculated += 3 * alpha * ex * x * x / (ex - 1) ** 2
        return cp_calculated * self.CONST_R

    def h_draw(self, parameters, temperature):
        h_calculated = 0.
        if len(parameters) % 2 != 0:
            print("Error!\n")
        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            h_calculated += 3 * alpha * x / (np.exp(x) - 1) * temperature

        return h_calculated * self.CONST_R

    def dh_draw(self, parameters, temperature):
        return self.h_draw(parameters, temperature) - self.h_draw(parameters, self.initial_temperature)

    def dh_cost(self, parameters, temperature, experiment):
        return (self.h_draw(parameters, temperature) - self.h_draw(parameters, 298.15) - experiment) / experiment

    def cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.dh_cost(parameters, h_temp, h_experiment), self.cp_cost(parameters, cp_temp, cp_experiment)
        return np.concatenate((dh, cp), axis=0)
