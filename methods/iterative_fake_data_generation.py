from dataframe import DataFrame, SingleDataFrame
from methods.fit_method import FitMethod
import numpy as np
import copy


class IterativeFakeDataGeneration(FitMethod):
    def __init__(self, iterations, c_mode_fitter, h_mode_fitter):
        self.c_mode_fitter = c_mode_fitter
        self.h_mode_fitter = h_mode_fitter
        self.last_fitted_model = None
        self.name = f"IterativeFakeData {iterations} iterations using {c_mode_fitter.name} and {h_mode_fitter.name}"
        self.current_fake_data = []
        self.iterations = iterations
        self.iteration_coefficients = []

    def fit(self, data_frame: DataFrame):
        """Implement iterative fit with fake data generation:
        1. Fit Cp data
        2. Interpolate dH data from Cp fit
        3. Add interpolated data to original dH
        4. Repeat for dH"""
        self.data_frame = data_frame
        self.enthalpy_temperature = data_frame.enthalpy_data.temperature
        self.heat_capacity_temperature = data_frame.heat_capacity_data.temperature
        self.enthalpy_data = data_frame.enthalpy_data.experiment
        self.iteration_coefficients = []

        self.c_mode_fitter.fit(data_frame)
        self.iteration_coefficients.append(self.c_mode_fitter.fit_coefficients)
        for iteration in range(0, self.iterations):
            fake_dh_data_frame = copy.deepcopy(self.data_frame)
            self.current_fake_data = SingleDataFrame(
                data_set=np.lib.recfunctions.merge_arrays((
                    np.array(self.heat_capacity_temperature, dtype=[("temperature", np.float64)]),
                    np.array(
                        self.c_mode_fitter.delta_enthalpy(self.c_mode_fitter.fit_coefficients,
                                                          self.heat_capacity_temperature),
                        dtype=[("experiment", np.float64)]))),
                data_type="dh",
                source_name=f"Fake dH data iteration #{iteration}")
            fake_dh_data_frame.enthalpy_data.append(self.current_fake_data)
            self.h_mode_fitter.fit(fake_dh_data_frame)
            # self.iteration_coefficients.append(self.h_mode_fitter.fit_coefficients)
            self.last_fitted_model = self.h_mode_fitter
            fake_cp_data_frame = copy.deepcopy(self.data_frame)
            self.current_fake_data = SingleDataFrame(
                data_set=np.lib.recfunctions.merge_arrays((
                    np.array(self.enthalpy_temperature, dtype=[("temperature", np.float64)]),
                    np.array(
                        self.h_mode_fitter.heat_capacity(self.h_mode_fitter.fit_coefficients,
                                                         self.enthalpy_temperature),
                        dtype=[("experiment", np.float64)]))),
                data_type="cp",
                source_name=f"Fake Cp data iteration #{iteration}")
            fake_cp_data_frame.heat_capacity_data.append(self.current_fake_data)
            self.c_mode_fitter.fit(fake_cp_data_frame)
            self.iteration_coefficients.append(self.c_mode_fitter.fit_coefficients)
            self.last_fitted_model = self.c_mode_fitter

        self.fit_enthalpy = self.last_fitted_model.fit_enthalpy
        self.fit_heat_capacity = self.last_fitted_model.fit_heat_capacity

        with open("iterations_output.tsv", "w") as f:
            f.write(f"iteration\t" + "\t".join([f"c_{i}" for i in range(len(self.iteration_coefficients[0]))]) + "\n")
            f.write(
                "\n".join(
                    [f"{i}\t" + "\t".join([f"{coefficient:.5f}" for coefficient in fit_coefficients])
                     for i, fit_coefficients in enumerate(self.iteration_coefficients)]
                ))

    def calculate_heat_capacity_residuals(self):
        self.heat_capacity_residuals = \
            (self.data_frame.heat_capacity_data.experiment - self.last_fitted_model.heat_capacity(
                self.last_fitted_model.fit_coefficients,
                self.data_frame.heat_capacity_data.temperature)) / \
            np.std(self.data_frame.heat_capacity_data.experiment - self.last_fitted_model.heat_capacity(
                self.last_fitted_model.fit_coefficients,
                self.data_frame.heat_capacity_data.temperature))

    def plot_heat_capacity(self, ax, **kwargs):
        """Plot heat capacity (derivative of the enthalpy fit result) using matplotlib."""
        try:
            ax.plot(self.data_frame.heat_capacity_data.temperature, self.last_fitted_model.heat_capacity(
                self.last_fitted_model.fit_coefficients,
                self.data_frame.heat_capacity_data.temperature), **kwargs)
        except ValueError:
            try:
                ax.plot(self.data_frame.heat_capacity_data.temperature, self.fit_heat_capacity, **kwargs)
            except ValueError:
                print('Something wrong with ', self.name)
                pass

    def get_fake_data(self):
        return self.current_fake_data
