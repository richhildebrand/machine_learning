from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer as SimpleImputer


def rph_graph(X, y, columns):
    my_model = GradientBoostingRegressor()
    regression_columns = columns
    my_imputer = SimpleImputer()
    X_regression = my_imputer.fit_transform(X)
    my_model.fit(X_regression, y)
    my_plots = plot_partial_dependence(my_model,       
                                    features=[0, 1, 2], # column numbers of plots we want to show
                                    X=X_regression,            # raw predictors data.
                                    feature_names=regression_columns, # labels on graphs
                                    grid_resolution=10) # number of values to plot on x axis