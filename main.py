import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor

from helpers.xgboost_helpers import rph_cross_validation
from helpers.xgboost_helpers import rph_find_column_to_drop

def get_columns_to_encode(candidate_train_predictors):
    low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                 candidate_train_predictors[cname].dtype == "object"]
    
    return low_cardinality_cols

def get_standard_columns(data):
    standard_columns = data.select_dtypes(exclude=['object']).columns
    
    columns_to_return = []
    for column in standard_columns:
        columns_to_return.append(column)
        
    return columns_to_return

data = pd.read_csv('./data/house_prices/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(columns=['SalePrice'])

columns_to_encode = get_columns_to_encode(X)
print("Columns to encode: " + str(columns_to_encode))

standard_columns = get_standard_columns(X)
print("Standard columns: " + str(standard_columns))

columns_to_keep = columns_to_encode + standard_columns
columns_to_keep.remove('Id')
X = data[columns_to_keep]
X = pd.get_dummies(X, columns=columns_to_encode)

#ensure encoded columns will match
final_test_data = pd.read_csv('./data/house_prices/test.csv')
final_test_X = final_test_data[columns_to_keep]

final_test_X = pd.get_dummies(final_test_X, columns=columns_to_encode)
X, final_test_X = X.align(final_test_X, join='inner', axis=1)

#find columns to remove
scores, model = rph_cross_validation(X, y)
base_mea = scores.mean()
print('starting mea ' + str(scores.mean()))


#columns_to_drop = rph_find_column_to_drop(X, y, standard_columns)
columns_to_drop = ['Fireplaces', 'GarageArea', 'MoSold', '1stFlrSF'] #16220.2109306
#columns_to_drop = ['Fireplaces', 'GarageArea', 'MoSold'] #16163.8201291
#columns_to_drop = ['Fireplaces', 'GarageArea', '1stFlrSF'] #16414.5766852
#columns_to_drop = ['Fireplaces', 'MoSold', '1stFlrSF'] #16393.2369589

print(columns_to_drop)
print(X.columns)
for column in columns_to_drop:
    X = X.drop(columns=[column])
    
scores, model = rph_cross_validation(X, y)
base_mea = scores.mean()
print('ending mea ' + str(scores.mean()))


#fun with graphs
my_model = GradientBoostingRegressor()
regression_columns = ['GarageCars', 'YrSold', 'LotArea']
my_imputer = SimpleImputer()
X_regression = my_imputer.fit_transform(X)
my_model.fit(X_regression, y)
my_plots = plot_partial_dependence(my_model,       
                                   features=[0, 1, 2], # column numbers of plots we want to show
                                   X=X_regression,            # raw predictors data.
                                   feature_names=regression_columns, # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis


#submit test data
print(columns_to_drop)
for column in columns_to_drop:
    final_test_X = final_test_X.drop(columns=[column])

my_imputer = SimpleImputer()
final_test_X  = my_imputer.fit_transform(final_test_X)

predicted_home_prices = model.predict(final_test_X)

my_submission = pd.DataFrame({'Id': final_test_data.Id, 'SalePrice': predicted_home_prices})
my_submission.to_csv('./data/house_prices/submission.csv', index=False)
print("CSV created")