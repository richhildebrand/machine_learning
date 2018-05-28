import pandas as pd

def rph_get_X_y_and_test_data(train_file_path, test_file_path, column_to_predict):
    train_data = pd.read_csv('./data/house_prices/train.csv')
    train_y = train_data[column_to_predict]
    train_X = train_data.drop(columns=[column_to_predict])

    test_data = pd.read_csv('./data/house_prices/test.csv')

    return train_y, train_X, test_data

def rph_drop_columns(data_frame, columns_to_drop):
    print('dropping columns -> ' + str(columns_to_drop))
    for column in columns_to_drop:
        data_frame = data_frame.drop(columns=[column])

    return data_frame

def rph_get_columns_to_encode(candidate_train_predictors):
    low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                 candidate_train_predictors[cname].dtype == "object"]
    

    print("\nColumns to encode: \n" + str(low_cardinality_cols) + '\n')
    return low_cardinality_cols

def rph_encode_columns(train_X, test_data, columns_to_encode):
    train_X = pd.get_dummies(train_X, columns=columns_to_encode)
    test_data = pd.get_dummies(test_data, columns=columns_to_encode)

    return train_X.align(test_data, join='inner', axis=1)


def rph_get_standard_columns(data):
    standard_columns = data.select_dtypes(exclude=['object']).columns
    
    columns_to_return = []
    for column in standard_columns:
        columns_to_return.append(column)
        
    print("\nNon object columns: \n" + str(columns_to_return) + '\n')
    return columns_to_return