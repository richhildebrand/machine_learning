import pandas as pd
from sklearn.preprocessing import Imputer as SimpleImputer

from helpers.pandas_helpers import rph_get_X_y_and_test_data
from helpers.pandas_helpers import rph_get_standard_columns
from helpers.pandas_helpers import rph_get_columns_to_encode
from helpers.pandas_helpers import rph_encode_columns
from helpers.pandas_helpers import rph_drop_columns

from helpers.xgboost_helpers import rph_cross_validation
from helpers.xgboost_helpers import rph_find_column_to_drop

from helpers.sklearn_helpers import rph_graph


train_file_path = './data/house_prices/train.csv'
test_file_path = './data/house_prices/test.csv'
column_to_predict = 'SalePrice'

y, X, final_test_data = rph_get_X_y_and_test_data(train_file_path, test_file_path, column_to_predict)

columns_to_encode = rph_get_columns_to_encode(X)
print("Columns to encode: " + str(columns_to_encode))

standard_columns = rph_get_standard_columns(X)
print("Standard columns: " + str(standard_columns))

columns_to_keep = columns_to_encode + standard_columns
columns_to_keep.remove('Id')
X = X[columns_to_keep]

#ensure encoded columns will match

X, final_test_X = rph_encode_columns(X, final_test_data, columns_to_encode)


#__start adjust model__
scores, model = rph_cross_validation(X, y)
print('starting mea ' + str(scores.mean()))

#columns_to_drop = rph_find_column_to_drop(X, y, standard_columns)
columns_to_drop = ['Fireplaces', 'GarageArea', 'MoSold', '1stFlrSF']
X = rph_drop_columns(X, columns_to_drop)

scores, model = rph_cross_validation(X, y)
print('ending mea ' + str(scores.mean()))
#__end adjust model__


rph_graph(X, y, ['GarageCars', 'YrSold', 'LotArea'])


#__start submit test data__
final_test_X = rph_drop_columns(final_test_X, columns_to_drop)

my_imputer = SimpleImputer()
final_test_X  = my_imputer.fit_transform(final_test_X)

predicted_home_prices = model.predict(final_test_X)

my_submission = pd.DataFrame({'Id': final_test_data.Id, 'SalePrice': predicted_home_prices})
my_submission.to_csv('./data/house_prices/submission.csv', index=False)
print("CSV created")
#__end submit test data__