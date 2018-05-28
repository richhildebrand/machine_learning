import pandas as pd

def rph_drop_columns(data_frame, columns_to_drop):
    print('dropping columns -> ' + str(columns_to_drop))
    for column in columns_to_drop:
        data_frame = data_frame.drop(columns=[column])

    return data_frame

def rph_get_columns_to_encode(candidate_train_predictors):
    low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                 candidate_train_predictors[cname].dtype == "object"]
    
    return low_cardinality_cols

def rph_get_standard_columns(data):
    standard_columns = data.select_dtypes(exclude=['object']).columns
    
    columns_to_return = []
    for column in standard_columns:
        columns_to_return.append(column)
        
    return columns_to_return