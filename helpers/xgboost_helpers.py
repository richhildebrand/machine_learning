from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.model_selection import cross_val_score

def rph_cross_validation(train_X, train_y):
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size = 0.30, random_state=1)

    my_imputer = SimpleImputer()
    train_X = my_imputer.fit_transform(train_X)
    test_X = my_imputer.transform(test_X)
    
    early_stopping_rounds = 30
    xgb_model = XGBRegressor(n_estimators=600, learning_rate=0.06)
    fit_params={'early_stopping_rounds': early_stopping_rounds, 
                'eval_metric': 'mae',
                'verbose': False,
                'eval_set': [[test_X, test_y]]}

    xgb_cv = cross_val_score(xgb_model, train_X, train_y, 
                             cv = 5, 
                             scoring = 'neg_mean_absolute_error',
                             fit_params = fit_params)
    
    xgb_model.fit(train_X, train_y, early_stopping_rounds=early_stopping_rounds, eval_set=[(test_X, test_y)], verbose=False)    
    return xgb_cv, xgb_model


def rph_find_column_to_drop(X, y, columns_to_check):
    columns_to_drop = []
    best_score = None
    for columnIndex in range(0, len(columns_to_check)):
        column = X.columns[columnIndex]
        train_X_copy = X.copy()
        train_X_copy = train_X_copy.drop(columns=[column])
        scores, model = rph_cross_validation(train_X_copy, y)
        adjusted_score = -1 * scores.mean()
        print('mea without ' + column + ' ' + str(adjusted_score))
        if not best_score or best_score > adjusted_score: 
            best_score = adjusted_score
            columns_to_drop = [column]
            
    return columns_to_drop
