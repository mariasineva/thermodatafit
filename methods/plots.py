import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()

method_colors = ['lightcoral', 'slategrey', 'palegreen', 'orangered', 'steelblue', 'thistle', 'orchid', 'olivedrab',
                 'gold', 'sienna']
dots_styles = ['^', '3', 'p', 'x', 'v', '2', 'o', '1', 's', 'H', 'd', '>', '<', 'D', '*', 'X', '4']
line_styles = ['-', '--', '-', '--', '-', '--', '-.', '-', '-']


def plot_fits(methods, subplot_position, subplot_title, data_frame):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    data_frame.dh_data.plot(s=20, color='navy')
    for (method, color, linestyle) in zip(methods, method_colors, line_styles):
        method.plot(ax, color=color, label=method.name, linestyle=linestyle)
    ax.set_title(subplot_title)

    ax.set_ylabel('dH, J/K/Mol')
    ax.set_xlabel('T, K')
    ax.legend(loc='upper left')


def plot_fit_derivatives(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color, linestyle) in zip(methods, method_colors, line_styles):
        method.plot_derivative(ax, color=color, label=method.name, linestyle=linestyle)
    ax.set_title(subplot_title)

    ax.set_xlabel('T, K')
    ax.legend(loc='upper left')


def plot_fit_derivatives_with_dots(methods, subplot_position, subplot_title, source):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    source.filter_by_temperature(min_temperature=300, max_temperature=3000)
    source.plot(s=20, color='navy')

    for (method, color, linestyle) in zip(methods, method_colors, line_styles):
        method.plot_derivative(ax, color=color, label=method.name, linestyle=linestyle)
    ax.set_title(subplot_title)

    ax.set_xlabel('T, K')
    ax.legend(loc='upper left')


def plot_residuals(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color) in zip(methods, method_colors):
        method.plot_residuals(ax, s=10, color=color, label=method.name)
    ax.set_title(subplot_title)
    ax.set_xlabel('T, K')
    ax.set_ylabel('Standardised residuals')
    ax.axhline(c='black')
    ax.legend(loc='upper left')


def plot_derivative_residuals(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color) in zip(methods, method_colors):
        method.plot_derivative_residuals(ax, s=10, color=color, label=method.name)
    ax.set_title(subplot_title)
    ax.set_xlabel('T, K')
    ax.set_ylabel('Standardised residuals')
    ax.axhline(c='black')
    ax.legend(loc='upper left')


def plot_normality(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color) in zip(methods, method_colors):
        method.plot_normality(ax, color=color, label=method.name)
    ax.set_title(subplot_title)
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Standardised residuals')
    ax.legend(loc='upper left')


def plot_leverage(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color) in zip(methods, method_colors):
        method.plot_leverage(ax, color=color, label=method.name)
    ax.set_title(subplot_title)
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardised residuals')
    ax.axhline(c='black')
    ax.legend(loc='upper right')


def plot_cooks_distance(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color) in zip(methods, method_colors):
        method.plot_cooks_distance(ax, color=color, label=method.name)
    ax.set_title(subplot_title)
    ax.set_xlabel('T,K')
    ax.set_ylabel('Cook\'s Distance')
    ax.axhline(y=4/len(methods[0].temp), c='black') #todo eliminate dh-only
    ax.legend(loc='upper left')


def plot_scale_location(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color) in zip(methods, method_colors):
        method.plot_scale_location(ax, color=color, label=method.name)
    ax.set_title(subplot_title)
    ax.set_xlabel('T, K')
    ax.set_ylabel('$\sqrt{|Standardized Residuals|}$')
    ax.legend(loc='upper left')


def plot_residuals_vs_fitted(methods, subplot_position, subplot_title):
    ax = plt.subplot(subplot_position)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for (method, color) in zip(methods, method_colors):
        method.plot_residuals_vs_fitted(ax, color=color, label=method.name)
    ax.set_title(subplot_title)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.legend(loc='upper right')


def basic_plots(methods, data, show_plot=True, save_plot=False, comment='', source_dots=False, mode='bigscreen'):
    if mode == 'laptop':
        plt.figure(figsize=(8, 8))
    else:
        plt.figure(figsize=(12, 12))

    plt.suptitle(f'{data.name} {comment}')

    plot_fits(methods, 221, data.name, data)
    if source_dots:
        plot_fit_derivatives_with_dots(methods, 222, 'Specific heat', source_dots)
    else:
        plot_fit_derivatives(methods, 222, 'Specific heat')
    plot_residuals(methods, 223, 'Residuals')
    plot_derivative_residuals(methods, 224, 'Heat Capacity Residuals')

    if show_plot:
        plt.tight_layout()
        plt.show()
    if save_plot:
        date = now.strftime('%d%m_%H%M')
        plt.savefig(f'../Plots/{data.name}_AllMethods_{date}.png', bbox_inches='tight')


def stats_plots(methods, data, show_plot=True, save_plot=False, comment='', mode='bigscreen'):
    if mode == 'laptop':
        plt.figure(figsize=(8, 8))
    else:
        plt.figure(figsize=(12, 6))
    # plt.figure(figsize=(12, 12))
    plt.suptitle(f'{data.name} statistics {comment}')

    # plot_residuals_vs_fitted(methods, 221, 'Residuals vs. Fit')
    # plot_normality(methods, 222, 'QQ')
    # plot_leverage(methods, 223, 'Leverage')
    # plot_scale_location(methods, 224, 'Scale – Location')

    plot_residuals_vs_fitted(methods, 121, 'Residuals vs. Fit')
    plot_normality(methods, 122, 'QQ')

    if show_plot:
        plt.show()
    if save_plot:
        date = now.strftime('%d%m_%H%M')
        plt.savefig(f'../Plots/{data.name}_AllMethodsStats_{date}.png', bbox_inches='tight')


def dh_and_cp_plots(methods, data, source_dots, show_plot=True, save_plot=False, comment=''):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'{data.name} {comment}')

    plot_fits(methods, 121, data.name, data)
    plot_fit_derivatives_with_dots(methods, 122, 'Specific heat', source_dots)
    if show_plot:
        plt.tight_layout()
        plt.show()
    if save_plot:
        date = now.strftime('%d%m_%H%M')
        plt.savefig(f'../Plots/dhcp/{data.name}_dh_and_cp_{date}.png', bbox_inches='tight')


def mad_trash_save_all(methods, data, comment=''):
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'{data.name} {comment}')

    plot_fits(methods, 111, data.name, data)
    plt.savefig(f'../Plots/Temp/{data.name}_AllMethods_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_fit_derivatives(methods, 111, 'Specific heat')
    plt.savefig(f'../Plots/Temp/{data.name}_Derivatives_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_residuals(methods, 111, 'Residuals')
    plt.savefig(f'../Plots/Temp/{data.name}_Residuals_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_derivative_residuals(methods, 111, 'Derivative Residuals')
    plt.savefig(f'../Plots/Temp/{data.name}_der_resid_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_cooks_distance(methods, 111, 'Outlier test')
    plt.savefig(f'../Plots/Temp/{data.name}_cooksd_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_residuals_vs_fitted(methods, 111, 'Residuals vs. Fit')
    plt.savefig(f'../Plots/Temp/{data.name}_resvsfit_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_normality(methods, 111, 'QQ')
    plt.savefig(f'../Plots/Temp/{data.name}_qq_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_leverage(methods, 111, 'Leverage')
    plt.savefig(f'../Plots/Temp/{data.name}_leverage_{comment}.png', bbox_inches='tight')
    plt.cla()

    plot_scale_location(methods, 111, 'Scale – Location')
    plt.savefig(f'../Plots/Temp/{data.name}_scalelocation_{comment}.png', bbox_inches='tight')
    plt.cla()
