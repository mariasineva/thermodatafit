from dataframe import DataFrame, SingleDataFrame
import methods.LS_constrained as lsq
import methods.LS_weighted_aux as wlsq
import methods.einstein_planck as nonlin
from methods.LS_joint import JointLeastSquares
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import kstest_normal
import methods.plots as draw
# from methods.weighted_cls import WPlainLeastSquares as wpls
from methods.LS_constrained_weighted import WeightedJointLeastSquares as wjls


def calculate_fits(methods, data):
    for method in methods:
        method.fit(data)


def calculate_residuals(methods):
    for method in methods:
        method.calculate_enthalpy_residuals()
        method.calculate_heat_capacity_residuals()


def calculate_info(methods):
    # for method in methods:
    #     print('\n' + method.name)
    #     method.calculate_refpoints()
    # print('\nR-squared')

    # for method in methods:
    #     print(method.get_rsquared())
    for method in methods:
        method.get_deviations()


def test_residuals_normality(methods):
    threshold_p = 0.05

    print("> Kolmogorov-Smirnov test")
    for method in methods:
        ksstat, pvalue = kstest_normal(method.residuals, dist='norm', pvalmethod='table')
        print(f"{method.name}: ks={ksstat}, p={pvalue}. {'❌' if pvalue < threshold_p else '✅'}.")


source_data = [
    # ['TuFr', 'W', 24.3068],
    ['allDataAlphadH', 'Ti_Alpha', 25.06, 'TiAlphaCp'],
    ['allDataBetadH', 'Ti_Beta', 25.06, 'TiBetaCp'],
    ['dataAu', 'Au', 25.27, 'AuCp'],
    ['UO2dHall', 'UO2', 15.2008, 'UO2CpCut'],
    ['VdHlow', 'V', 24.48, 'VCp'],
]

if __name__ == '__main__':
    show_plots = True
    # show_plots = False
    # save_plots = True
    save_plots = False

    # comment = '4 terms'
    min_power = -1
    max_power = 2
    # for i in range(len(source_data)):
    for i in [3]:
        fit_methods = [
            wlsq.WeightedLeastSquaresWithAuxiliaryFunction(power=1),
            lsq.СonstrainedLeastSquaresSM(min_power=-1, max_power=2),
            nonlin.EinsteinPlankSum(3, mode='h'), nonlin.EinsteinPlankSum(3, mode='c'),
            nonlin.EinsteinPlankSum(3, mode='j'),
            JointLeastSquares(min_power=-1, max_power=3, mode='h'),
            # JointLeastSquares(min_power=-1, max_power=3, mode='c'),
            # JointLeastSquares(min_power=-1, max_power=3, mode='cc'),
            # JointLeastSquares(min_power=-1, max_power=3, mode='j'),
            # wpls(min_power=-1, max_power=3),
            # wjls(min_power=-1, max_power=3, mode='j', weight_parameter=0.99),
            # wjls(min_power=-1, max_power=3, mode='j', weight_parameter=0.001),
            # wjls(min_power=-1, max_power=3, mode='c', weight_parameter=0.99),
            # wjls(min_power=-1, max_power=3, mode='cc'),
        ]
        data_file, data_name, c_ref, hc_file_name = source_data[i]
        data_dict = {'dh': f'Data/{data_file}.txt'}

        if hc_file_name != '':
            data_dict['cp'] = f'Data/{hc_file_name}.txt'
        data_frame = DataFrame(data_dict, name=data_name)
        data_frame.set_initial_conditions(reference_temperature=298.15, reference_heat_capacity_value=c_ref,
                                          reference_heat_capacity_error=0.1,
                                          experiment_weight=0.01)

        if hc_file_name != '':  # todo this looks like a duplicate of some sort
            hc_data = SingleDataFrame(f'Data/{hc_file_name}.txt', name=data_name + 'Cp')

        calculate_fits(fit_methods, data_frame)
        calculate_residuals(fit_methods)

        comment = ''
        screen_type = 'laptop'
        # screen_type = 'bigscreen'
        draw.basic_plots(fit_methods, data_frame, show_plots, save_plots, comment, hc_data, screen_type)
        # draw.stats_plots(fit_methods, data_frame, show_plots, save_plots, comment, screen_type)
        # todo stat plots for dh & cp : qq; res vs fit
        # draw.dh_and_cp_plots(fit_methods, data_frame, hc_data, show_plots, save_plots, comment)
        # draw.mad_trash_save_all(fit_methods, data_frame, comment)

        # calculate_info(fit_methods)
        # calculate_cp_difference(fit_methods, hc_data)
