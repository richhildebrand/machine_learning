import pandas as pd
from sklearn.preprocessing import Imputer as SimpleImputer

from helpers.pandas_helpers import rph_get_X_y_and_test_data
from helpers.pandas_helpers import rph_get_standard_columns
from helpers.pandas_helpers import rph_get_columns_to_encode
from helpers.pandas_helpers import rph_encode_columns
from helpers.pandas_helpers import rph_drop_columns
from helpers.pandas_helpers import rph_create_output_file

from helpers.xgboost_helpers import rph_cross_validation
from helpers.xgboost_helpers import rph_find_non_object_column_to_drop
from helpers.xgboost_helpers import rph_find_encoded_column_to_drop

from helpers.sklearn_helpers import rph_graph


train_file_path = './data/house_prices/train.csv'
test_file_path = './data/house_prices/test.csv'
output_file_path = './data/house_prices/submission.csv'
column_to_predict = 'SalePrice'
id_column = 'Id'

y, X, test_data = rph_get_X_y_and_test_data(train_file_path, test_file_path, column_to_predict)

columns_to_encode = rph_get_columns_to_encode(X, 20)
print(len(columns_to_encode))

standard_columns = rph_get_standard_columns(X)
columns_to_keep = columns_to_encode + standard_columns

X = X[columns_to_keep]

#rph_find_encoded_column_to_drop(y, X, test_data, columns_to_encode, id_column)
#columns_to_drop = rph_find_non_object_column_to_drop(X, y, standard_columns)

#encoded_columns_to_drop = ['CentralAir']
encoded_columns_to_drop = ['CentralAir', 'GarageType']
for column in encoded_columns_to_drop: columns_to_encode.remove(column)


columns_to_drop = ['Fireplaces', 'GarageArea', 'MoSold', '1stFlrSF'] + encoded_columns_to_drop
X = rph_drop_columns(X, columns_to_drop)
test_data = rph_drop_columns(test_data, columns_to_drop)

X, test_data = rph_encode_columns(X, test_data, columns_to_encode)
X = X.drop(columns=id_column)

scores, model = rph_cross_validation(X, y)
print('ending mea ' + str(scores.mean()))


rph_create_output_file(model, test_data, id_column, column_to_predict, output_file_path)