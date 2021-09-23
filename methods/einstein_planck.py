from dataframe import DataFrame
from methods.fit_method import FitMethod
import numpy as np
from scipy.optimize import least_squares as scipy_ls
from scipy.stats import t as students_t


class EinsteinPlankMethod(FitMethod):
    """list of functions"""

    def __init__(self):
        self.CONST_R = 8.3144598
        self.initial_params = [0.01, 1.00]
        self.initial_boundaries = [[0.0, 0.0], [500.0, 1.0e5]]
        self.initial_temperature = 298.15
        self.fit_coefficients = self.initial_params
        self.bounds = self.initial_boundaries

    def heat_capacity_cost(self, parameters, temperature, experiment):
        cp_calculated = 0.0
        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            ex = np.exp(x)
            cp_calculated += 3 * alpha * ex * x * x / (ex - 1) ** 2

        return (cp_calculated * self.CONST_R - experiment) / experiment

    def enthalpy_cost(self, parameters, temperature, experiment):
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

    def heat_capacity(self, parameters, temperature):
        cp_calculated = 0.0
        for alpha, theta in zip(parameters[::2], parameters[1::2]):
            x = theta / temperature
            ex = np.exp(x)
            cp_calculated += 3 * alpha * ex * x * x / (ex - 1) ** 2
        return cp_calculated * self.CONST_R

    def enthalpy(self, parameters, temperature):
        h_calculated = 0.
        if len(parameters) % 2 != 0:
            print("Error!\n")
        for i in range(0, len(parameters), 2):
            alpha = parameters[i]
            theta = parameters[i + 1]
            x = theta / temperature
            h_calculated += 3 * alpha * x / (np.exp(x) - 1) * temperature

        return h_calculated * self.CONST_R

    def delta_enthalpy(self, parameters, temperature):
        return self.enthalpy(parameters, temperature) - self.enthalpy(parameters, self.initial_temperature)

    def delta_enthalpy_cost(self, parameters, temperature, experiment):
        return (self.enthalpy(parameters, temperature) - self.enthalpy(parameters, 298.15) - experiment) / experiment

    def joint_cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.delta_enthalpy_cost(parameters, h_temp, h_experiment), \
                 self.heat_capacity_cost(parameters, cp_temp, cp_experiment)
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
        self.fit_coefficients = self.initial_params
        self.bounds = self.initial_boundaries

        self.confidence_params = [0.0, 0.0]

    def level_t(self):
        return students_t.isf(0.975, len(self.data_frame.temp) - len(self.fit_coefficients) / 2)

    def approx(self):
        for i in range(0, self.power - 1):
            self.fit_coefficients.append(0.01)
            self.fit_coefficients.append(1.00)
            self.bounds[0].append(0.0)
            self.bounds[0].append(0.0)
            self.bounds[1].append(500.0)
            self.bounds[1].append(1.0e5)

            if self.mode == 'j':
                res_lsq = scipy_ls(self.joint_cost_function, self.fit_coefficients, bounds=self.bounds,
                                   args=(
                                       self.data_frame.enthalpy_data.temperature,
                                       self.data_frame.enthalpy_data.experiment,
                                       self.data_frame.heat_capacity_data.temperature,
                                       self.data_frame.heat_capacity_data.experiment))
            elif self.mode == 'h':
                res_lsq = scipy_ls(self.delta_enthalpy_cost, self.fit_coefficients, bounds=self.bounds,
                                   args=(
                                       self.data_frame.enthalpy_data.temperature,
                                       self.data_frame.enthalpy_data.experiment))
            elif self.mode == 'c':
                res_lsq = scipy_ls(self.heat_capacity_cost, self.fit_coefficients, bounds=self.bounds,
                                   args=(self.data_frame.heat_capacity_data.temperature,
                                         self.data_frame.heat_capacity_data.experiment))
            else:
                print('Wrong calculation type!')
                raise ValueError('Type can only be h c j.\n')

        self.fit_coefficients = res_lsq.x.tolist()

    def covariance_stuff(self, cost_function):
        ls_zero_step = scipy_ls(cost_function, self.fit_coefficients,
                                args=(self.data_frame.temp, self.data_frame.experiment))
        temporary_parameters = ls_zero_step.x
        jacobian = ls_zero_step.jac

        covariance_matrix = np.linalg.inv(jacobian.T.dot(jacobian))

        temporary_fit = cost_function(temporary_parameters, self.data_frame.temp, self.data_frame.experiment)
        temporary_fit *= temporary_fit
        # todo read https://stackoverflow.com/questions/28702631/scipy-curve-fit-returns-negative-variance
        sigma_squared = sum(temporary_fit) / (len(self.data_frame.temp) - len(self.fit_coefficients) / 2)
        covariance_matrix *= sigma_squared

        diagonal = np.sqrt(np.diagonal(covariance_matrix))
        self.confidence_params = np.sqrt(diagonal)
        self.confidence_params *= np.abs(self.level_t())
        print('Confidence something: ', self.confidence_params)

        need_new = False
        for a, b in zip(self.fit_coefficients, self.confidence_params):
            if a < b:
                need_new = True
        if need_new:
            self.fit_coefficients = np.append(self.fit_coefficients, 0.1)
            self.fit_coefficients = np.append(self.fit_coefficients, 10)

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.approx()

        self.enthalpy_temperature = data_frame.enthalpy_data.temperature
        self.heat_capacity_temperature = data_frame.heat_capacity_data.temperature

        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.enthalpy_data.temperature)

        if not len(self.data_frame.heat_capacity_data.temperature):
            self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.enthalpy_data.temperature)
        else:
            self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.heat_capacity_temperature)

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        experiment_cp = hc_data.experiment
        cp_temp = hc_data.temp
        fit_cp = self.heat_capacity(self.fit_coefficients, cp_temp)
        return experiment_cp - fit_cp

    def calculate_enthalpy_residuals(self):
        # if self.mode == 'h':
        self.enthalpy_residuals = \
            (self.data_frame.enthalpy_data.experiment - self.fit_enthalpy) / np.std(self.data_frame.enthalpy_data.experiment - self.fit_enthalpy)
        # elif self.mode == 'c':
        #     self.residuals = (self.data_frame.cp_e - self.fit_derivative) / \
        #                      np.std(self.data_frame.cp_e - self.fit_derivative)
        # elif self.mode == 'j':
        #    self.residuals = np.concatenate((
        #        (self.data_frame.dh_e - self.fit) / np.std(self.data_frame.dh_e - self.fit),
        #        (self.data_frame.cp_e - self.fit_derivative) / np.std(self.data_frame.cp_e - self.fit_derivative)),
        #        axis=0)

    def calculate_heat_capacity_residuals(self):
        self.heat_capacity_residuals = \
            (self.data_frame.heat_capacity_data.experiment - self.fit_heat_capacity) / np.std(self.data_frame.heat_capacity_data.experiment - self.fit_heat_capacity)

    def plot_enthalpy_residuals(self, ax, **kwargs):
        """Plot standartised residuals using matplotlib."""
        # if self.mode == 'h':
        ax.scatter(self.data_frame.enthalpy_data.temperature, self.enthalpy_residuals, **kwargs)
        # elif self.mode == 'c':
        #     ax.scatter(self.data_frame.cp_t, self.residuals, **kwargs)
        # elif self.mode == 'j':
        #     ax.scatter(np.concatenate((self.data_frame.dh_t, self.data_frame.cp_t), axis=0), self.residuals, **kwargs)

    def result_txt_output(self):
        return {
            "substance": self.data_frame.name,
            "method": self.name,
            "coefficients": self.fit_coefficients,
            "function type": "polynomial",
            "min temperature": self.enthalpy_temperature[0],
            "max temperature": self.enthalpy_temperature[-1]
        }


class EinsteinAndPolynomialCorrection(FitMethod):
    """list of functions"""

    def __init__(self, power, mode='h'):
        self.name = "E-P. and correction term: %s mode" % mode
        self.CONST_R = 8.3144598
        self.initial_params = [0.01, 1.00, 1.00, 1.00, 1.00, ]
        # self.initial_boundaries = [[-10.0, 100.0], [-10.0, 100.0]]
        self.initial_temperature = 298.15
        self.params = self.initial_params
        # self.bounds = self.initial_boundaries
        self.mode = mode

    def heat_capacity_cost(self, parameters, temperature, experiment):
        cp_calculated = 0.0
        theta = parameters[0]
        a = parameters[1]
        b = parameters[2]
        c = parameters[3]
        d = parameters[4]
        t = temperature
        theta_t = theta / temperature
        e_theta = np.exp(theta_t)
        cp_calculated += 3 * e_theta * theta_t * theta_t / (
                e_theta - 1) ** 2 * self.CONST_R + a * t + b * t * t + c * t ** 3 + d * t ** 4
        return (cp_calculated - experiment) / experiment

    def enthalpy_cost(self, parameters, temperature, experiment):
        h_calculated = 0.0
        theta = parameters[0]
        a = parameters[1]
        b = parameters[2]
        c = parameters[3]
        d = parameters[4]
        t = temperature
        theta_t = theta / temperature
        e_theta = np.exp(theta_t)
        h_calculated += 3 / 2 * self.CONST_R * theta_t * (e_theta + 1) / (
                e_theta - 1) * temperature + a * t ** 2 / 2 + b * t ** 3 / 3 + c * t ** 4 / 4 + d * t ** 5 / 5

        return (h_calculated - experiment) / experiment

    def heat_capacity(self, parameters, temperature):
        cp_calculated = 0.0  # todo combine with cost
        theta = parameters[0]
        a = parameters[1]
        b = parameters[2]
        c = parameters[3]
        d = parameters[4]
        t = temperature
        theta_t = theta / temperature
        e_theta = np.exp(theta_t)
        cp_calculated += 3 * e_theta * theta_t * theta_t / (
                e_theta - 1) ** 2 * self.CONST_R + a * t + b * t * t + c * t ** 3 + d * t ** 4
        return cp_calculated

    def enthalpy(self, parameters, temperature):
        h_calculated = 0.0
        theta = parameters[0]
        a = parameters[1]
        b = parameters[2]
        c = parameters[3]
        d = parameters[4]
        t = temperature
        theta_t = theta / temperature
        e_theta = np.exp(theta_t)
        h_calculated += 3 / 2 * self.CONST_R * theta_t * (e_theta + 1) / (
                e_theta - 1) * temperature + a * t ** 2 / 2 + b * t ** 3 / 3 + c * t ** 4 / 4 + d * t ** 5 / 5

        return h_calculated

    def delta_enthalpy(self, parameters, temperature):
        return self.enthalpy(parameters, temperature) - self.enthalpy(parameters, self.initial_temperature)

    def delta_enthalpy_cost(self, parameters, temperature, experiment):
        return (self.enthalpy(parameters, temperature) - self.enthalpy(parameters, 298.15) - experiment) / experiment

    def joint_cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.delta_enthalpy_cost(parameters, h_temp, h_experiment), self.heat_capacity_cost(parameters,
                                                                                                     cp_temp,
                                                                                                     cp_experiment)
        return np.concatenate((dh, cp), axis=0)

    def approx(self):

        if self.mode == 'j':
            res_lsq = scipy_ls(self.joint_cost_function, self.params,  # bounds=self.bounds,
                               args=(self.data_frame.dh_t, self.data_frame.dh_e,
                                     self.data_frame.cp_t, self.data_frame.cp_e))
        elif self.mode == 'h':
            res_lsq = scipy_ls(self.delta_enthalpy_cost, self.params,  # bounds=self.bounds,
                               args=(self.data_frame.dh_t, self.data_frame.dh_e))
        elif self.mode == 'c':
            res_lsq = scipy_ls(self.heat_capacity_cost, self.params,  # bounds=self.bounds,
                               args=(self.data_frame.cp_t, self.data_frame.cp_e))
        else:
            print('Wrong calculation type!')
            raise ValueError('Type can only be h c j.\n')
        # print(res_lsq.x.tolist())
        self.params = res_lsq.x.tolist()

    def fit(self, data_frame: DataFrame):

        self.data_frame = data_frame

        self.approx()

        self.enthalpy_temperature = data_frame.dh_t

        self.heat_capacity_temperature = data_frame.cp_t

        self.fit_enthalpy = self.delta_enthalpy(self.params, self.data_frame.dh_t)

        if not len(self.data_frame.cp_t):
            self.fit_heat_capacity = self.heat_capacity(self.params, self.data_frame.dh_t)
        else:
            self.fit_heat_capacity = self.heat_capacity(self.params, self.heat_capacity_temperature)

    def calculate_heat_capacity_residuals(self):
        self.heat_capacity_residuals = \
            (self.data_frame.cp_e - self.fit_heat_capacity) / np.std(self.data_frame.cp_e - self.fit_heat_capacity)

    def result_txt_output(self):
        return {
            "substance": self.data_frame.name,
            "method": self.name,
            "coefficients": self.params,
            "function type": "polynomial",
            "min temperature": self.enthalpy_temperature[0],
            "max temperature": self.enthalpy_temperature[-1]
        }
