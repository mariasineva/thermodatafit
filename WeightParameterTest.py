from dataframe import DataFrame, SingleDataFrame
import methods.non_linear as nonlin
import methods.plots as draw
from methods.weighted_cls import WeightedJointLeastSquares as wjls
from methods.joint_lsq import JointLeastSquares, PlainJointLeastSquares


def calculate_fits(methods, data):
    for method in methods:
        method.fit(data)


def calculate_residuals(methods):
    for method in methods:
        method.calculate_residuals()
        method.calculate_heat_capacity_residuals()


source_data = [
    ['test_dh_curve', 'test_data', 0.1, 'test_cp_line'],
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
    h_w = 10 ** (-6)
    c_w = 100.0
    for i in [0]:
        fit_methods = [
            # wjls(min_power=-1, max_power=2, mode='h', weight_parameter=0.1, cp_weight=1.0, dh_weight=1.0),
            wjls(min_power=-1, max_power=2, mode='c', weight_parameter=0.01, cp_weight=1.0, dh_weight=0.0),
            # wjls(min_power=-1, max_power=2, mode='j', weight_parameter=0.01, cp_weight=c_w, dh_weight=h_w),
            # wjls(min_power=-1, max_power=2, mode='j', weight_parameter=0.01, cp_weight=h_w, dh_weight=c_w),
            # wjls(min_power=-1, max_power=3, mode='j', weight_parameter=0.0001),
            # wjls(min_power=-1, max_power=3, mode='j_relative_error'),
            # JointLeastSquares(min_power=0, max_power=2, mode='h'),
            JointLeastSquares(min_power=-1, max_power=2, mode='c'),
            # JointLeastSquares(min_power=-1, max_power=2, mode='cc'),
        ]
        data_file, data_name, c_ref, hc_file_name = source_data[i]
        data_dict = {'dh': f'Data/{data_file}.txt'}
        # data_dict = {}
        if hc_file_name != '':
            data_dict['cp'] = f'Data/{hc_file_name}.txt'
        data_frame = DataFrame(data_dict, name=data_name)
        data_frame.set_initial_conditions(reference_temperature=0.1, reference_cvalue=c_ref, reference_cerror=0.1,
                                          experiment_weight=0.01)

        if hc_file_name != '':
            hc_data = SingleDataFrame(f'Data/{hc_file_name}.txt', name=data_name + 'Cp')

        calculate_fits(fit_methods, data_frame)
        #
        comment = ''
        # draw.basic_plots(fit_methods, data_frame, show_plots, save_plots, comment)
        screen_type = 'laptop'
        # screen_type = 'bigscreen'
        # draw.basic_plots(fit_methods, data_frame, show_plots, save_plots, comment, hc_data, screen_type)
        draw.dh_and_cp_plots(fit_methods, data_frame, hc_data, show_plots, save_plots, comment)
        # draw.stats_plots(fit_methods, data_frame, show_plots, save_plots, comment, screen_type)

