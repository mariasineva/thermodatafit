from dataframe import DataFrame, SingleDataFrame
import methods.least_squares as lsq
import methods.weighted_aux_ls as wlsq
import methods.non_linear as nonlin
from methods.joint_lsq import JointLeastSquares, PlainJointLeastSquares
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import kstest_normal
import methods.plots as draw
# from methods.weighted_cls import WPlainLeastSquares as wpls
from methods.weighted_cls import WeightedJointLeastSquares as wjls


def calculate_fits(methods, data):
    for method in methods:
        method.fit(data)


def calculate_residuals(methods):
    for method in methods:
        method.calculate_enthalpy_residuals()
        method.calculate_heat_capacity_residuals()


def calculate_info(methods):
    # print('Reference points: f(t0), f\'(t0)')
    # for method in methods:
    #     print('\n' + method.name)
    #     method.calculate_refpoints()
    # print('\nR-squared')

    # for method in methods:
    #     print(method.get_rsquared())
    for method in methods:
        # print('\n' + method.name)
        method.get_deviations()


def calculate_cp_difference(methods, hc_data):
    print('Cp difference')
    with open('report_cp.txt', 'w') as f:

        for method in methods:
            print('\n' + method.name)
            f.write('\n' + method.name)
            for item in method.compare_to_cp(hc_data):
                f.write("%s\n" % item)
            print(method.compare_to_cp(hc_data))


def test_residuals_normality(methods):
    threshold_p = 0.05

    print("> Kolmogorov-Smirnov test")
    for method in methods:
        ksstat, pvalue = kstest_normal(method.residuals, dist='norm', pvalmethod='table')
        print(f"{method.name}: ks={ksstat}, p={pvalue}. {'❌' if pvalue < threshold_p else '✅'}.")


source_data = [
    # ['TuFr', 'W', 24.3068],
    ['allDataAlphadH', 'Ti_Alpha', 25.06, 'TiAlphaCp'],
    # ['allDataBetadH', 'Ti_Beta', 25.06, 'TiBetaCp'],
    ['dataAu', 'Au', 25.27, 'AuCp'],
    ['UO2dHall', 'UO2', 15.2008, 'UO2Cp3k'],
    ['VdHlow', 'V', 24.48, 'VCp'],
]

if __name__ == '__main__':
    show_plots = True
    # show_plots = False
    # save_plots = True
    save_plots = False

    # comment = '4 terms'
    # for i in range(len(source_data)):
    for i in [3]:
        fit_methods = [
            wlsq.WeightedLeastSquaresWithAuxiliaryFunction(power=1),
            wlsq.WeightedLeastSquaresWithAuxiliaryFunction(power=2),
            # lsq.СonstrainedLeastSquaresSM(min_power=-1, max_power=2),
            # lsq.СonstrainedLeastSquaresSM(min_power=-1, max_power=3),
            # nonlin.EinsteinPlankSum(2),
            # wlsq.WeightedLeastSquaresWithAuxiliaryFunction(power=3),
            # lsq.СonstrainedLeastSquaresSM(min_power=-1, max_power=4),
            # nonlin.EinsteinPlankSum(3, mode='h'),
            # nonlin.EinsteinPlankSum(3, mode='c'),
            # nonlin.EinsteinPlankSum(3, mode='j'),
            # lsq.LeastSquaresFitNoFree(),
            # lsq.OrdinaryLeastSquaresSM(power=3),
            # wlsq.WeightedСonstrainedLeastSquaresSM(min_power=-1, max_power=2),
            # lad.LeastAbsoluteFit(power=2, initial_coefficients=[1.0, 1.0, 1.0]),
            # JointLeastSquares(min_power=-1, max_power=3, mode='h'),
            # JointLeastSquares(min_power=-1, max_power=3, mode='c'),
            # JointLeastSquares(min_power=-1, max_power=3, mode='cc'),
            # JointLeastSquares(min_power=-1, max_power=3, mode='j'),
            # PlainJointLeastSquares(min_power=-1, max_power=3),
            # wpls(min_power=-1, max_power=3),
            # wjls(min_power=-1, max_power=3),
            # wjls(min_power=-1, max_power=3, mode='h'),
            # wjls(min_power=-1, max_power=3, mode='c'),
            # wjls(min_power=-1, max_power=3, mode='cc'),
            # wjls(min_power=-1, max_power=3, mode='j', weight_parameter=0.1),
            # wjls(min_power=-1, max_power=3, mode='j', weight_parameter=0.01),
            # wjls(min_power=-1, max_power=3, mode='j'),
            # wjls(min_power=-1, max_power=3, mode='j', weight_parameter=0.0001),
            # wjls(min_power=-1, max_power=3, mode='j_relative_error'),
        ]
        data_file, data_name, c_ref, hc_file_name = source_data[i]
        data_dict = {'dh': f'Data/{data_file}.txt'}
        # data_dict = {}
        if hc_file_name != '':
            data_dict['cp'] = f'Data/{hc_file_name}.txt'
        data_frame = DataFrame(data_dict, name=data_name)
        data_frame.set_initial_conditions(reference_temperature=298.15, reference_heat_capacity_value=c_ref,
                                          reference_heat_capacity_error=0.1,
                                          experiment_weight=0.01)

        if hc_file_name != '':
            hc_data = SingleDataFrame(f'Data/{hc_file_name}.txt', name=data_name + 'Cp')

        # data_frame.filter_by_temperature(min_temperature=439, max_temperature=1156)

        # data_frame.filter_outliers_by_cooks_distance()
        # comment = 'cooks_distance_filter'
        #
        # data_frame.filter_outliers_by_residual(2.5)
        # comment = 'residual_filter'

        # data_frame.filter_outliers_by_dffits()
        # comment = 'dffits'

        calculate_fits(fit_methods, data_frame)
        calculate_residuals(fit_methods)
        #
        comment = ''
        # draw.basic_plots(fit_methods, data_frame, show_plots, save_plots, comment)
        screen_type = 'laptop'
        # screen_type = 'bigscreen'
        draw.basic_plots(fit_methods, data_frame, show_plots, save_plots, comment, hc_data, screen_type)
        # draw.stats_plots(fit_methods, data_frame, show_plots, save_plots, comment, screen_type)
# todo stat plots for dh & cp : qq; res vs fit 
#         draw.dh_and_cp_plots(fit_methods, data_frame, hc_data, show_plots, save_plots, comment)
        # draw.mad_trash_save_all(fit_methods, data_frame, comment)

        # calculate_info(fit_methods)

        # calculate_cp_difference(fit_methods, hc_data)

# todo weighted CLS;
