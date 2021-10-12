import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

from dataframe import DataFrame
from methods.fit_method import FitMethod


class IterativeDataDeletion(FitMethod):
    def __init__(self, iterations, fitter):
        self.fitter = fitter
        self.last_fitted_model = None
        self.name = f"IterativeDataDeletion {fitter.__class__.__name__}"
        self.iterations = iterations
        self.iteration_coefficients = []

    def fit(self, data_frame: DataFrame):
        """Implement iterative fit with deleting data points with maximal residuals."""
        self.data_frame = data_frame
        self.enthalpy_temperature = data_frame.enthalpy_data.temperature
        self.heat_capacity_temperature = data_frame.heat_capacity_data.temperature
        self.enthalpy_data = data_frame.enthalpy_data.experiment
        self.iteration_coefficients = []

        for iteration in range(0, self.iterations):
            self.fitter.fit(data_frame)
            self.fit_enthalpy = self.fitter.fit_enthalpy
            self.iteration_coefficients.append(self.fitter.fit_coefficients)
            self.filter_outliers()

        with open("iterations_output.tsv", "w") as f:
            f.write(f"iteration\t" + "\t".join([f"c_{i}" for i in range(len(self.iteration_coefficients[0]))]) + "\n")
            f.write(
                "\n".join(
                    [f"{i}\t" + "\t".join([f"{coefficient:.5f}" for coefficient in fit_coefficients])
                     for i, fit_coefficients in enumerate(self.iteration_coefficients)]
                ))

    def animate_fit(self, data_frame: DataFrame):
        self.data_frame = data_frame
        self.enthalpy_temperature = data_frame.enthalpy_data.temperature
        self.heat_capacity_temperature = data_frame.heat_capacity_data.temperature
        self.enthalpy_data = data_frame.enthalpy_data.experiment
        self.iteration_coefficients = []
        fig = plt.figure(figsize=(16, 8))
        camera = Camera(fig)

        ax = plt.subplot(121)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('T, K')
        self.data_frame.enthalpy_data.plot(s=20, color="navy")

        ax = plt.subplot(122)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('T, K')
        self.data_frame.heat_capacity_data.plot(s=20, color="navy")
        camera.snap()

        for iteration in range(0, self.iterations):
            self.fitter.fit(data_frame)
            self.iteration_coefficients.append(self.fitter.fit_coefficients)

            ax = plt.subplot(121)
            ax.scatter(
                self.data_frame.enthalpy_data.original_temperature,
                self.data_frame.enthalpy_data.original_experiment,
                s=10, color="crimson")
            self.data_frame.enthalpy_data.plot(s=20, color="navy")
            self.fitter.plot_enthalpy(ax, color="navy", linestyle="-")

            ax = plt.subplot(122)
            ax.scatter(
                self.data_frame.heat_capacity_data.original_temperature,
                self.data_frame.heat_capacity_data.original_experiment,
                s=10, color="crimson")
            self.data_frame.heat_capacity_data.plot(s=20, color="navy")
            self.fitter.plot_heat_capacity(ax, color="navy", linestyle="-")

            camera.snap()

            self.filter_outliers()

        camera.animate().save("filtering_iterations.gif")

    def filter_outliers(self, threshold=3.0):
        self.calculate_enthalpy_residuals()
        sigma_dh = np.std(self.enthalpy_residuals)
        is_not_dh_outlier = [abs(residual) < threshold * sigma_dh for residual in self.enthalpy_residuals]
        self.data_frame.enthalpy_data.temperature = self.data_frame.enthalpy_data.temperature[is_not_dh_outlier]
        self.data_frame.enthalpy_data.experiment = self.data_frame.enthalpy_data.experiment[is_not_dh_outlier]

        self.calculate_heat_capacity_residuals()
        sigma_cp = np.std(self.heat_capacity_residuals)
        is_not_cp_outlier = [abs(residual) < threshold * sigma_cp for residual in self.heat_capacity_residuals]
        self.data_frame.heat_capacity_data.temperature = \
            self.data_frame.heat_capacity_data.temperature[is_not_cp_outlier]
        self.data_frame.heat_capacity_data.experiment = self.data_frame.heat_capacity_data.experiment[is_not_cp_outlier]

    def calculate_heat_capacity_residuals(self):
        self.heat_capacity_residuals = \
            (self.data_frame.heat_capacity_data.experiment - self.fitter.heat_capacity(
                self.fitter.fit_coefficients,
                self.data_frame.heat_capacity_data.temperature)) / \
            np.std(self.data_frame.heat_capacity_data.experiment - self.fitter.heat_capacity(
                self.fitter.fit_coefficients,
                self.data_frame.heat_capacity_data.temperature))

    def calculate_enthalpy_residuals(self):
        self.enthalpy_residuals = \
            (self.data_frame.enthalpy_data.experiment - self.fitter.delta_enthalpy(
                self.fitter.fit_coefficients,
                self.data_frame.enthalpy_data.temperature)) / \
            np.std(self.data_frame.enthalpy_data.experiment - self.fitter.delta_enthalpy(
                self.fitter.fit_coefficients,
                self.data_frame.enthalpy_data.temperature))

    def plot_heat_capacity(self, ax, **kwargs):
        """Plot heat capacity (derivative of the enthalpy fit result) using matplotlib."""
        try:
            ax.plot(self.data_frame.heat_capacity_data.temperature, self.fitter.heat_capacity(
                self.fitter.fit_coefficients,
                self.data_frame.heat_capacity_data.temperature), **kwargs)
        except ValueError:
            try:
                ax.plot(self.data_frame.heat_capacity_data.temperature, self.fit_heat_capacity, **kwargs)
            except ValueError:
                print('Something wrong with ', self.name)
                pass
