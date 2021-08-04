from dataframe import DataFrame, SingleDataFrame
from methods.LS_constrained import СonstrainedLeastSquaresSM
from methods.LS_constrained_weighted import WeightedJointLeastSquares
from methods.LS_joint import JointLeastSquares
from methods.LS_weighted_aux import WeightedLeastSquaresWithAuxiliaryFunction
from methods.einstein_planck import EinsteinPlankSum, EinsteinAndPolynomialCorrection
from methods.external_curves import ExternalCurves
import methods.plots as draw


def calculate_fits(methods, data):
    for method in methods:
        method.fit(data)


def calculate_residuals(methods):
    for method in methods:
        method.calculate_enthalpy_residuals()
        method.calculate_heat_capacity_residuals()


def calculate_info(methods):
    for method in methods:
        mean, least, maximal = method.get_deviations()


SOURCE_DATA = [
    # [data_file, data_name, c_ref, hc_file_name]
    # ['allDataAlphadH', 'Ti_Alpha', 25.06, 'TiAlphaCp'],
    # ['allDataBetadH', 'Ti_Beta', 25.06, 'TiBetaCp'],
    ['dataAu', 'Au', 25.27, 'AuCp'],
    # ['UO2dHall', 'UO2', 15.2008, 'UO2CpCut'],
    # ['VdHlow', 'V', 24.48, 'VCp'],
    # ['NpO2', 'NpO2', 66.0, 'NpO2CpCut150'],
]

FIT_METHODS = [
    ExternalCurves.create_from_cp_params(
        source_name='Königs 2014', min_power=-2, max_power=3, min_temp=1, max_temp=10,
        cp_coefficients=[-0.78932 * 1e6, 0, 64.7712, 43.8574 * 1e-3, -35.0695 * 1e-6, 13.1917 * 1e-9]),
    # WeightedLeastSquaresWithAuxiliaryFunction(power=2),
    # СonstrainedLeastSquaresSM(min_power=-1, max_power=2),
    # EinsteinPlankSum(3, mode='h'),
    # EinsteinPlankSum(3, mode='c'),
    EinsteinPlankSum(3, mode='j'),
    EinsteinAndPolynomialCorrection(3, mode='c'),
    # JointLeastSquares(min_power=-1, max_power=3, mode='h'),
    JointLeastSquares(min_power=-1, max_power=4, mode='c'),
    # JointLeastSquares(min_power=-1, max_power=3, mode='cc'),
    # JointLeastSquares(min_power=-1, max_power=4, mode='j'),
    # WeightedJointLeastSquares(min_power=-1, max_power=3, mode='j', weight_parameter=0.99),
    # WeightedJointLeastSquares(min_power=-1, max_power=3, mode='cc'),
]

if __name__ == '__main__':
    SHOW_PLOTS = True
    SAVE_PLOTS = False
    SCREEN_TYPE = 'laptop'  # laptop or bigscreen
    COMMENT = ''

    for data_file, data_name, c_ref, hc_file_name in SOURCE_DATA:
        data_dict = {'dh': f'Data/{data_file}.txt'}
        if hc_file_name != '':
            data_dict['cp'] = f'Data/{hc_file_name}.txt'

        data_frame = DataFrame.from_sources_dict(data_dict, name=data_name)
        data_frame.set_initial_conditions(reference_heat_capacity_value=c_ref)
        # data_frame.export_to_json_file(f'{data_name}.json')
        # data_frame.export_to_table_view(f'{data_name}_check.txt')

        if hc_file_name != '':  # todo this looks like a duplicate of some sort
            hc_data = SingleDataFrame.from_txt_file(f'Data/{hc_file_name}.txt', data_type='cp', name=f'{data_name}Cp')

        calculate_fits(FIT_METHODS, data_frame)
        calculate_residuals(FIT_METHODS)

        draw.basic_plots(FIT_METHODS, data_frame, SHOW_PLOTS, SAVE_PLOTS, COMMENT, hc_data, SCREEN_TYPE)
        # draw.stats_plots(FIT_METHODS, data_frame, show_plots, SHOW_PLOTS, COMMENT, SCREEN_TYPE)
        # todo stat plots for dh & cp : qq; res vs fit
        # draw.dh_and_cp_plots(FIT_METHODS, data_frame, hc_data, SHOW_PLOTS, SAVE_PLOTS, COMMENT)
        # draw.mad_trash_save_all(FIT_METHODS, data_frame, COMMENT)

        # calculate_info(FIT_METHODS)
