import copy
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class SingleDataFrame:
    def __init__(self, filename: str, name: str = "Unnamed data set"):
        """Create data frame from txt file."""
        self.name = name

        data_set = np.loadtxt(filename, dtype=[('temp', 'f8'), ('experiment', 'f8')])
        np.sort(data_set, order='temp')
        self.original_temperature = data_set['temp']
        self.original_experiment = data_set['experiment']

        self.set_initial_conditions()
        self.reset_filters()
        # self.type

    def set_initial_conditions(self, reference_temperature=298.15, reference_enthalpy_value=0.0, 
                               reference_heat_capacity_value=25.27,
                               reference_heat_capacity_error=0.1, experiment_weight=0.01):
        """Set initial conditions for experimental data in this data frame."""
        self.reference_temperature = reference_temperature
        self.reference_enthalpy_value = reference_enthalpy_value
        self.reference_heat_capacity_value = reference_heat_capacity_value
        self.reference_heat_capacity_error = reference_heat_capacity_error
        self.experiment_weight = experiment_weight

    def reset_filters(self):
        self.temperature = copy.deepcopy(self.original_temperature)
        self.experiment = copy.deepcopy(self.original_experiment)

    def filter_by_temperature(self, min_temperature, max_temperature):
        min_index = np.searchsorted(self.temperature, min_temperature)
        max_index = np.searchsorted(self.temperature, max_temperature)
        self.temperature = self.temperature[min_index:max_index]
        self.experiment = self.experiment[min_index:max_index]

    def filter_outliers_by_cooks_distance(self, power=3):
        source_matrix = np.column_stack(
            [self.temperature ** i if i != 0 else np.ones(len(self.temperature)) for i in range(0, power + 1)])
        ols_fit = sm.OLS(self.experiment, source_matrix).fit()
        influence = 4 / len(self.temperature)
        cooks_distance = ols_fit.get_influence().cooks_distance[0]
        is_not_outlier = [distance < influence for distance in cooks_distance]

        is_outlier = [distance > influence for distance in cooks_distance]
        print('Log:\n', cooks_distance[is_outlier], '\n', self.temperature[is_outlier], '\n')

        self.temperature = self.temperature[is_not_outlier]
        self.experiment = self.experiment[is_not_outlier]

    def filter_outliers_by_residual(self, threshold=3.0, power=3):
        source_matrix = np.column_stack(
            [self.temperature ** i if i != 0 else np.ones(len(self.temperature)) for i in range(-1, power + 1)])
        ols_fit = sm.OLS(self.experiment, source_matrix).fit()
        fit = np.dot(source_matrix, ols_fit.params)

        residuals = (self.experiment - fit) / np.std(self.experiment - fit)

        is_not_outlier = [abs(res) < threshold for res in residuals]
        is_outlier = [abs(res) >= threshold for res in residuals]

        print('Log:\n', residuals[is_outlier], '\n', self.temperature[is_outlier], '\n')
        self.temperature = self.temperature[is_not_outlier]
        self.experiment = self.experiment[is_not_outlier]

    def filter_outliers_by_dffits(self, power=3):

        threshold = 2 * np.sqrt(4 / len(self.temperature))
        source_matrix = np.column_stack(
            [self.temperature ** i if i != 0 else np.ones(len(self.temperature)) for i in range(-1, power + 1)])
        ols_fit = sm.OLS(self.experiment, source_matrix).fit()

        dffit = ols_fit.get_influence().dffits[0]

        is_not_outlier = [abs(res) < threshold for res in dffit]

        self.temperature = self.temperature[is_not_outlier]
        self.experiment = self.experiment[is_not_outlier]

    def filter_outliers_by_hats(self, power=3):

        threshold = 2 * np.sqrt(4 / len(self.temperature))
        source_matrix = np.column_stack(
            [self.temperature ** i if i != 0 else np.ones(len(self.temperature)) for i in range(-1, power + 1)])
        ols_fit = sm.OLS(self.experiment, source_matrix).fit()

        hats = ols_fit.get_influence().hat_matrix_diag[0]
        #todo дописать тут

        # is_not_outlier = [abs(res) < threshold for res in dffit]
        # self.temperature = self.temperature[is_not_outlier]
        # self.experiment = self.experiment[is_not_outlier]

    def plot(self, **kwargs):
        """Plot data using matplotlib."""
        plt.scatter(self.temperature, self.experiment, **kwargs)


class DataFrame:
    def __init__(self, sources: dict, name: str = "Unnamed data set"):
        """Create data frame from txt file."""
        self.name = name
        self.sources = sources
        self.set_initial_conditions()

        try:
            self.heat_capacity_data = SingleDataFrame(sources['cp'], self.name)

        except KeyError as no_cp_key:
            print('No heat capacity data! ', no_cp_key)
            self.cp_t, self.cp_e = [], []
        else:
            self.cp_t, self.cp_e = self.heat_capacity_data.temperature, self.heat_capacity_data.experiment

        try:
            self.enthalpy_data = SingleDataFrame(sources['dh'], self.name)
        except KeyError as no_dh_key:
            print('No enthalpy data! ', no_dh_key)
            self.dh_t, self.dh_e = [], []
        else:
            self.dh_t, self.dh_e = self.enthalpy_data.original_temperature, self.enthalpy_data.original_experiment

    def set_initial_conditions(self, reference_temperature=298.15, reference_enthalpy_value=0.0,
                               reference_heat_capacity_value=25.27,
                               reference_heat_capacity_error=0.1, experiment_weight=0.01):
        """Set initial conditions for experimental data in this data frame."""
        self.reference_temperature = reference_temperature
        self.reference_enthalpy_value = reference_enthalpy_value
        self.reference_heat_capacity_value = reference_heat_capacity_value
        self.reference_heat_capacity_error = reference_heat_capacity_error
        self.experiment_weight = experiment_weight


