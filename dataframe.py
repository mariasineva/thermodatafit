import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class SingleDataFrame:
    """Data frame for a single experiment."""

    def __init__(self, data_set: np.ndarray, data_type: str, source_name: str = "Unnamed data set", year: int = 0,
                 reference_temperature: float = 298.15, reference_enthalpy_value: float = 0.0,
                 reference_heat_capacity_value: float = 25.27, reference_heat_capacity_error: float = 0.1,
                 experiment_weight: float = 0.01):
        # TODO(mariasineva): проверять, что в датасете есть поля temperature и experiment и они одинаковой длины.
        self.data_set = data_set
        self.original_temperature = data_set["temperature"]
        self.original_experiment = data_set["experiment"]
        self.temperature = copy.deepcopy(self.original_temperature)
        self.experiment = copy.deepcopy(self.original_experiment)

        self.data_type = data_type
        self.source_name = source_name
        self.year = year
        self.reference_temperature = reference_temperature
        self.reference_enthalpy_value = reference_enthalpy_value
        self.reference_heat_capacity_value = reference_heat_capacity_value
        self.reference_heat_capacity_error = reference_heat_capacity_error
        self.experiment_weight = experiment_weight

    @staticmethod
    def from_txt_file(filename: str, data_type: str, name: str = "Unnamed data set"):
        """Create data frame from a txt file."""
        data_set = np.loadtxt(filename, dtype=[("temperature", "f8"), ("experiment", "f8")])
        np.sort(data_set, order="temperature")

        return SingleDataFrame(data_set, data_type=data_type, source_name=name)

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
        print("Log:\n", cooks_distance[is_outlier], "\n", self.temperature[is_outlier], "\n")

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

        print("Log:\n", residuals[is_outlier], "\n", self.temperature[is_outlier], "\n")
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
        # todo дописать тут

        # is_not_outlier = [abs(res) < threshold for res in dffit]
        # self.temperature = self.temperature[is_not_outlier]
        # self.experiment = self.experiment[is_not_outlier]

    def apply_filter(self, filter_mode, **kwargs):
        {
            "temperature": self.filter_by_temperature,
            "cooks_distance": self.filter_outliers_by_cooks_distance,
            "residual": self.filter_outliers_by_residual,
            "dffits": self.filter_outliers_by_dffits
        }[filter_mode](**kwargs)

    def plot(self, **kwargs):
        """Plot data using matplotlib."""
        plt.scatter(self.temperature, self.experiment, **kwargs)


class DataFrame:
    def __init__(self, substance: str, heat_capacity_data: SingleDataFrame,
                 enthalpy_data: SingleDataFrame, reference_temperature: float = 298.15,
                 reference_enthalpy_value: float = 0.0, reference_heat_capacity_value: float = 25.27,
                 reference_heat_capacity_error: float = 0.1, experiment_weight: float = 0.01):
        self.substance = substance
        self.name = substance
        self.reference_temperature = reference_temperature
        self.reference_enthalpy_value = reference_enthalpy_value
        self.reference_heat_capacity_value = reference_heat_capacity_value
        self.reference_heat_capacity_error = reference_heat_capacity_error
        self.experiment_weight = experiment_weight

        self.heat_capacity_data = heat_capacity_data
        self.enthalpy_data = enthalpy_data
        self.cp_t = heat_capacity_data.temperature
        self.cp_e = heat_capacity_data.experiment
        self.dh_t = enthalpy_data.temperature
        self.dh_e = enthalpy_data.experiment

    @staticmethod
    def from_sources_dict(sources: dict, name: str = "Unnamed data set"):
        heat_capacity_data = SingleDataFrame.from_txt_file(sources["cp"], data_type="cp", name=name) \
            if "cp" in sources else \
            SingleDataFrame(np.array([], dtype=[("temperature", "f8"), ("experiment", "f8")]), data_type="cp")
        enthalpy_data = SingleDataFrame.from_txt_file(sources["dh"], data_type="dh", name=name) \
            if "dh" in sources else \
            SingleDataFrame(np.array([], dtype=[("temperature", "f8"), ("experiment", "f8")]), data_type="dh")

        return DataFrame(substance=name, heat_capacity_data=heat_capacity_data, enthalpy_data=enthalpy_data)

    @staticmethod
    def from_json_file(json_file_name: str):
        with open(json_file_name) as f:
            json_data = json.loads(f.read())

            single_data_frames = []
            for data_source in json_data["data_sources"]:
                data_set = np.lib.recfunctions.merge_arrays(
                    (
                        np.array(data_source["temperature"], dtype=[("temperature", np.float64)]),
                        np.array(data_source[data_source["data_type"]], dtype=[("experiment", np.float64)])
                    ),
                    usemask=False, asrecarray=False)
                single_data_frames.append(
                    SingleDataFrame(data_set,
                                    data_source["data_type"],
                                    data_source["source_name"],
                                    data_source["year"],
                                    json_data["reference_temperature"],
                                    json_data["reference_enthalpy_value"],
                                    json_data["reference_heat_capacity_value"],
                                    json_data["reference_heat_capacity_error"],
                                    json_data["experiment_weight"]))

            heat_capacity_data = next(
                (single_data_frame for single_data_frame in single_data_frames if single_data_frame.data_type == "cp"),
                SingleDataFrame(np.array([], dtype=[("temperature", "f8"), ("experiment", "f8")]), data_type="cp"))
            enthalpy_data = next(
                (single_data_frame for single_data_frame in single_data_frames if single_data_frame.data_type == "dh"),
                SingleDataFrame(np.array([], dtype=[("temperature", "f8"), ("experiment", "f8")]), data_type="dh"))

            return DataFrame(json_data["substance"],
                             heat_capacity_data,
                             enthalpy_data,
                             json_data["reference_temperature"],
                             json_data["reference_enthalpy_value"],
                             json_data["reference_heat_capacity_value"],
                             json_data["reference_heat_capacity_error"],
                             json_data["experiment_weight"])

    def set_initial_conditions(self, reference_temperature=298.15, reference_enthalpy_value=0.0,
                               reference_heat_capacity_value=25.27,
                               reference_heat_capacity_error=0.1, experiment_weight=0.01):
        """Set initial conditions for experimental data in this data frame."""
        self.reference_temperature = reference_temperature
        self.reference_enthalpy_value = reference_enthalpy_value
        self.reference_heat_capacity_value = reference_heat_capacity_value
        self.reference_heat_capacity_error = reference_heat_capacity_error
        self.experiment_weight = experiment_weight
