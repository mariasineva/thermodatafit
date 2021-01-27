from dataframe import DataFrame
from methods.fit_method import FitMethod
import numpy as np
from scipy.optimize import least_squares as scipy_ls


class JointLeastSquares(FitMethod):
    def __init__(self, min_power, max_power, mode='h'):
        assert min_power <= 0, "min power should be <= 0"
        assert max_power >= 1, "max power should be >= 1"
        assert max_power - min_power > 1, "there should be at least 3 powers for regression"

        self.mode = mode
        self.name = f"Joint LS (powers {min_power} ... {max_power}), {mode} mode"
        self.min_power = min_power
        self.max_power = max_power
        self.params = np.ones(self.max_power - self.min_power - 1)
        # self.params = [1.1, 1.2, 1.3]

    def enthalpy(self, parameters, temperature):
        '''final enthalpy function, uses all coefficients after calculating a0 and a1'''
        # assert len(parameters) == len(self.params), "Wrong parameters count for h_draw"
        h_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            h_calculated += temperature ** i * parameter
        return h_calculated

    def enthalpy_initial_function(self, parameters, temperature):
        '''f(T) = H(T) - H(0), calculated for initial set of parameters
        (before converting to stationary coefficients),
        ignoring params[0] and params[1] which are obtained with initial values'''

        assert len(parameters) == len(self.params), "Wrong parameters count for enthalpy"

        t_ref = self.data_frame.reference_temperature
        h_calculated = 0.0

        powers = [x for x in range(self.min_power, self.max_power + 1) if x not in [0, 1]]
        for i, parameter in zip(powers, parameters):
            h_calculated += (temperature ** i + (i - 1) * t_ref ** i - i * temperature * t_ref ** (i - 1)) \
                            * parameter
        return h_calculated

    def delta_enthalpy(self, parameters, temperature):
        return self.enthalpy(parameters, temperature) - self.enthalpy(parameters, self.data_frame.reference_temperature)

    def heat_capacity(self, parameters, temperature):
        cp_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            if i not in [0]:
                cp_calculated += i * temperature ** (i - 1) * parameter
        return cp_calculated

    def stationary_coefficients(self, initial_coefs, t_ref, c_0, c_1):
        a_1 = c_1 - \
              sum([i * initial_coefs[i - self.min_power] * t_ref ** (i - 1)
                   for i in range(self.min_power, 0)]) - \
              sum([i * initial_coefs[i - 2 - self.min_power] * t_ref ** (i - 1)
                   for i in range(2, self.max_power + 1)])
        a_0 = c_0 - a_1 * t_ref - sum([initial_coefs[i - self.min_power] * t_ref ** i
                                       for i in range(self.min_power, 0)]) - \
              sum([initial_coefs[i - 2 - self.min_power] * t_ref ** i
                   for i in range(2, self.max_power + 1)])

        fit_coefficients = []
        fit_coefficients.extend(initial_coefs[:-self.min_power])
        fit_coefficients.extend([a_0, a_1])
        fit_coefficients.extend(initial_coefs[-self.min_power:])
        return fit_coefficients

    def delta_enthalpy_initial_function(self, parameters, temperature):
        '''H(T) - H(T_ref)'''
        return self.enthalpy_initial_function(parameters, temperature) \
               - self.enthalpy_initial_function(parameters, self.data_frame.reference_temperature)

    def delta_enthalpy_constrained_cost(self, parameters, temperature, experiment):
        # renormalize cost
        residuals = (self.delta_enthalpy_initial_function(parameters, temperature) - experiment) / experiment
        print(self.name, "e_dh:", sum(residuals ** 2))
        return residuals

    def heat_capacity_initial_function(self, parameters, temperature):
        cp_calculated = 0.0
        t_ref = self.data_frame.reference_temperature
        powers = [x for x in range(self.min_power, self.max_power + 1) if x not in [0, 1]]
        for i, parameter in zip(powers, parameters):
            cp_calculated += i * (temperature ** (i - 1) - t_ref ** (i - 1)) * parameter
        return cp_calculated

    def heat_capacity_cost(self, parameters, temperature, experiment):
        # residuals = self.cp_draw(parameters, temperature) - experiment
        # renormalize cost
        residuals = (self.heat_capacity(parameters, temperature) - experiment) / experiment
        # print(self.name, "e_cp:", sum(residuals ** 2))
        return residuals

    def heat_capacity_constrained_cost(self, parameters, temperature, experiment):
        residuals = self.heat_capacity_initial_function(parameters, temperature) - experiment
        # print(self.name, 'e_cp:', sum(residuals ** 2))
        return residuals

    def joint_cost_function(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.delta_enthalpy_constrained_cost(parameters, h_temp, h_experiment), \
                 self.heat_capacity_constrained_cost(parameters, cp_temp, cp_experiment)
        # print(self.name, "e_dh:", sum(dh ** 2), "e_cp:", sum(cp ** 2), "e_dh+cp:",
        #       sum(np.concatenate((dh ** 2, cp ** 2), axis=0)))
        return np.concatenate((dh, cp), axis=0)

    def h_mode_approximation(self):
        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.enthalpy_temperature))

        c_0 = self.data_frame.reference_enthalpy_value
        c_1 = self.data_frame.reference_heat_capacity_value

        updated_experiment = self.data_frame.dh_e - c_0 * np.ones(len(self.enthalpy_temperature)) - c_1 * (
                self.enthalpy_temperature - t_ref_vector)

        initial_fit = scipy_ls(self.delta_enthalpy_constrained_cost, self.params, args=(self.data_frame.dh_t, updated_experiment))
        print(self.name, initial_fit.x.tolist())

        self.fit_coefficients = self.stationary_coefficients(initial_fit.x.tolist(), t_ref, c_0, c_1)
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def c_mode_approximation(self):
        self.params = np.ones(self.max_power - self.min_power + 1)
        initial_fit = scipy_ls(self.heat_capacity_cost, self.params, args=(self.data_frame.cp_t, self.data_frame.cp_e))

        self.fit_coefficients = initial_fit.x.tolist()
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def c_mode_constrained_approximation(self):
        t_ref = self.data_frame.reference_temperature

        c_0 = self.data_frame.reference_enthalpy_value
        c_1 = self.data_frame.reference_heat_capacity_value

        updated_cp = self.data_frame.cp_e - c_1 * np.ones(len(self.heat_capacity_temperature))

        initial_fit = scipy_ls(self.heat_capacity_constrained_cost, self.params, args=(self.data_frame.cp_t, updated_cp))

        self.fit_coefficients = self.stationary_coefficients(initial_fit.x.tolist(), t_ref, c_0, c_1)
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def j_mode_approximation(self):
        t_ref = self.data_frame.reference_temperature
        t_ref_vector = t_ref * np.ones(len(self.enthalpy_temperature))

        c_0 = self.data_frame.reference_enthalpy_value
        c_1 = self.data_frame.reference_heat_capacity_value

        updated_enthalpy = self.enthalpy_data - c_0 * np.ones(len(self.enthalpy_temperature)) - c_1 * (
                self.enthalpy_temperature - t_ref_vector)
        updated_heat_capacity = self.data_frame.cp_e - c_1 * np.ones(len(self.heat_capacity_temperature))

        initial_fit = scipy_ls(self.joint_cost_function, self.params,
                               args=(self.enthalpy_temperature, updated_enthalpy, self.data_frame.cp_t, updated_heat_capacity))

        self.fit_coefficients = self.stationary_coefficients(initial_fit.x.tolist(), t_ref, c_0, c_1)
        self.fit_enthalpy = self.delta_enthalpy(self.fit_coefficients, self.data_frame.dh_t)
        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.cp_t)

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.enthalpy_temperature = data_frame.dh_t
        self.heat_capacity_temperature = data_frame.cp_t
        self.enthalpy_data = data_frame.dh_e

        {
            'cc': self.c_mode_constrained_approximation,
            'c': self.c_mode_approximation,
            'j': self.j_mode_approximation,
            'h': self.h_mode_approximation
        }[self.mode]()

        print('Joint lsq result:', self.name, self.fit_coefficients)

    # def calculate_enthalpy_residuals(self):
    #     if self.mode == 'h':
        # self.enthalpy_residuals = (self.data_frame.dh_e - self.fit_enthalpy) / np.std(
        #     self.data_frame.dh_e - self.fit_enthalpy)

    def calculate_heat_capacity_residuals(self):
        self.heat_capacity_residuals = (self.data_frame.cp_e - self.heat_capacity(self.fit_coefficients,
                                                                                  self.heat_capacity_temperature)) / \
                                       np.std(self.data_frame.cp_e - self.heat_capacity(self.fit_coefficients,
                                                                                        self.heat_capacity_temperature))


class PlainJointLeastSquares(FitMethod):
    def __init__(self, min_power, max_power):
        self.name = 'Plain Joint LS'
        self.min_power = min_power
        self.max_power = max_power
        self.params = np.ones(self.max_power - self.min_power + 1)

    def enthalpy(self, parameters, temperature):
        h_calculated = 0.0
        for i, k in zip(range(self.min_power, self.max_power + 1), range(len(self.params))):
            h_calculated += temperature ** i * parameters[k]
        return h_calculated

    def delta_enthalpy(self, parameters, temperature):
        return self.enthalpy(parameters, temperature) - self.enthalpy(parameters, self.data_frame.reference_temperature)

    def delta_enthalpy_cost(self, parameters, temperature, experiment):
        return self.enthalpy(parameters, temperature) - experiment

    def heat_capacity(self, parameters, temperature):
        cp_calculated = 0.0
        powers = [x for x in range(self.min_power, self.max_power + 1)]
        for i, parameter in zip(powers, parameters):
            cp_calculated += i * temperature ** (i - 1) * parameter
        return cp_calculated

    def heat_capacity_cost(self, parameters, temperature, experiment):
        return self.heat_capacity(parameters, temperature) - experiment

    def joint_cost(self, parameters, h_temp, h_experiment, cp_temp, cp_experiment):
        dh, cp = self.delta_enthalpy_cost(parameters, h_temp, h_experiment), self.heat_capacity_cost(parameters, cp_temp, cp_experiment)
        return np.concatenate((dh, cp), axis=0)

    def fit(self, data_frame: DataFrame):
        self.data_frame = data_frame

        self.enthalpy_temperature = data_frame.dh_t
        self.heat_capacity_temperature = data_frame.cp_t
        self.enthalpy_data = data_frame.dh_e

        # initial_fit = scipy_ls(self.dh_cost, self.params,  # bounds=self.bounds,
        #                    args=(self.data_frame.dh_t, self.experiment))
        #
        # initial_fit = scipy_ls(self.cost_function, self.params,  # bounds=self.bounds,
        #                    args=(self.data_frame.dh_t, self.data_frame.dh_e,
        #                    self.data_frame.cp_t, self.data_frame.cp_e))

        initial_fit = scipy_ls(self.heat_capacity_cost, self.params,  # bounds=self.bounds,
                           args=(self.heat_capacity_temperature, self.data_frame.cp_e))

        self.fit_coefficients = initial_fit.x.tolist()

        self.enthalpy_fit = self.delta_enthalpy(self.fit_coefficients, self.enthalpy_temperature)

        self.fit_heat_capacity = self.heat_capacity(self.fit_coefficients, self.data_frame.dh_t)

    def calculate_enthalpy_residuals(self):
        # if self.mode == 'h':
        self.enthalpy_residuals = (self.data_frame.dh_e - self.enthalpy_fit) / np.std(self.data_frame.dh_e - self.enthalpy_fit)
        # elif self.mode == 'c':
        #     self.residuals = (self.data_frame.cp_e - self.fit_derivative) / \
        #                      np.std(self.data_frame.cp_e - self.fit_derivative)
        # elif self.mode == 'j':
        #    self.residuals = np.concatenate((
        #        (self.data_frame.dh_e - self.fit) / np.std(self.data_frame.dh_e - self.fit),
        #        (self.data_frame.cp_e - self.fit_derivative) / np.std(self.data_frame.cp_e - self.fit_derivative)),
        #        axis=0)

    def calculate_heat_capacity_residuals(self):
        self.heat_capacity_residuals = (self.data_frame.cp_e - self.heat_capacity(self.fit_coefficients, self.heat_capacity_temperature)) / \
                                    np.std(self.data_frame.cp_e - self.heat_capacity(self.fit_coefficients, self.heat_capacity_temperature))
