from dataframe import DataFrame
import numpy as np
from statsmodels.graphics.gofplots import ProbPlot
import seaborn as sns


class FitMethod:
    def __init__(self):
        self.name = "Unnamed fit method"

    def fit(self, data_frame: DataFrame):
        """Fit experimental data in given data frame."""
        pass

    def plot(self, ax, **kwargs):
        """Plot fit result using matplotlib."""
        ax.plot(self.temp, self.fit, **kwargs)

    def plot_derivative(self, ax, **kwargs):
        """Plot derivative of the fit result using matplotlib."""
        try:
            ax.plot(self.data_frame.cp_t, self.fit_derivative, **kwargs)
        except ValueError:
            try:
                ax.plot(self.data_frame.dh_t, self.fit_derivative, **kwargs)
            except ValueError:
                print('Something wrong with ', self.name)
                pass

    def calculate_residuals(self):
        self.residuals = (self.experiment - self.fit) / np.std(self.experiment - self.fit)

    def calculate_derivative_residuals(self):
        pass

    def calculate_refpoints(self):
        """Calculate values of f(t0) and f'(t0)"""
        pass

    def annotate_residuals(self, ax):
        """Find max 3 residuals and annotate their dots on QQ plot"""
        pass

    def annotate_leverage(self, ax):
        """Explicitly annotate points with highest cook's distance"""
        pass

    def annotate_cooks_distance(self, ax):
        """Explicitly annotate points with influential cook's distance"""
        pass

    def plot_residuals(self, ax, **kwargs):
        """Plot standartised residuals using matplotlib."""
        ax.scatter(self.temp, self.residuals, **kwargs)

    def plot_derivative_residuals(self, ax, **kwargs):
        """Plot standartised residuals using matplotlib."""
        ax.scatter(self.cp_temp, self.derivative_residuals, **kwargs)

    def plot_normality(self, ax, color, label):
        """qq normality test"""
        qq = ProbPlot(self.residuals)
        qq.qqplot(line='45', alpha=0.5, color=color, lw=0.5, ax=ax, label=label)

        self.annotate_residuals(ax)

    def plot_leverage(self, ax, color, label):
        """Residuals vs Leverage"""
        try:
            leverage = self.aux_fit.get_influence().hat_matrix_diag
            sns.regplot(leverage, self.residuals, scatter=False, ci=False, lowess=True,
                        line_kws={'color': color, 'lw': 1, 'alpha': 0.8})
            ax.scatter(leverage, self.residuals, color=color, label=label)
            self.annotate_leverage(ax)
        except AttributeError:
            pass

    def plot_scale_location(self, ax, color, label):
        """Scale-Location"""
        try:
            residuals_sqrt = np.sqrt(np.abs(self.residuals))
            ax.scatter(self.data_frame.temp, residuals_sqrt, color=color, label=label)
            sns.regplot(self.data_frame.temp, residuals_sqrt, scatter=False, ci=False, lowess=True,
                        line_kws={'color': color, 'lw': 1, 'alpha': 0.8})
        except AttributeError:
            pass

    def plot_cooks_distance(self, ax, color, label):
        """Cook's Distance plot"""
        try:
            cooks_distance = self.aux_fit.get_influence().cooks_distance[0]
            ax.scatter(self.data_frame.temp, cooks_distance, color=color, label=label)
            sns.regplot(self.data_frame.temp, cooks_distance, scatter=False, ci=False, lowess=True,
                        line_kws={'color': color, 'lw': 1, 'alpha': 0.8})
            self.annotate_cooks_distance(ax)
        except AttributeError:
            pass

    def plot_residuals_vs_fitted(self, ax, color, label):
        """Residuals vs fitted"""
        try:
            sns.residplot(self.fit, self.data_frame.dh_e, lowess=True,
                          scatter_kws={'alpha': 0.5}, label=label, color=color, line_kws={'lw': 1, 'alpha': 0.8})
        except AttributeError:
            pass

    def get_rsquared(self):
        """Return R-squared """
        mean = 1 / len(self.data_frame.experiment) * sum(self.data_frame.experiment)
        rss = sum([(y - y_hat) ** 2 for y, y_hat in zip(self.data_frame.experiment, self.fit)])
        tss = sum([(y - mean) ** 2 for y in self.data_frame.experiment])

        return 1 - rss / tss
        # return np.sqrt(rss)/len(self.data_frame.experiment) # this is norm sum of sq of resids

    def get_deviations(self):
        """Return R-squared """
        mean = 1 / len(self.data_frame.experiment) * np.sqrt(sum(self.residuals ** 2))
        least = np.min(np.abs(self.residuals))
        maximal = np.max(np.abs(self.residuals))

        print(mean, least, maximal)
        return mean, least, maximal

    def compare_to_cp(self, hc_data):
        """Compare to cp"""
        pass
