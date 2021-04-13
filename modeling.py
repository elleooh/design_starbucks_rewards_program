import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from bayes_opt import BayesianOptimization

def return_normalize_entropy(predicted, target):
    """
    Calculate normalized entropy based on predicted results and target

    INPUT:
    predicted: pandas series of predicted results
    target: pandas series of target labels

    OUTPUT:
    normalized entropy (evaluation metric)
    """
    
    try:
        if len(predicted) != len(target):
            return 'lengths not equal!'
    except:
        target = target.get_label() # to be used in lgb Bayesian optimization
    
    N = len(target)
    p = np.mean(target)
    
    numerator = -(1/N)*sum([target[i]*np.log(predicted[i]) + 
                              (1-target[i])*np.log(1-predicted[i])
                              for i in range(N)])
    denominator = -(p*np.log(p)+(1-p)*np.log(1-p))
    
    return 'normalized_entropy', numerator/denominator, False

# Hybrid GBDT (LightGBM) & LR model
def bayes_parameter_opt_lgb(x_train, y_train, init_round=15, opt_round=25, n_folds=5 \
    , random_seed=0):
    """
    Apply bayesian optimization for LightGBM

    INPUT:
    X: feature dataframe
    y: label

    OUTPUT:
    dictionary of optimal hyperparameters
    """

    # prepare data
    train_data = lgb.Dataset(data=x_train, label=y_train, free_raw_data=False)

    # parameters
    def lgb_eval(n_estimators, learning_rate, num_leaves, feature_fraction \
        , max_depth , min_split_gain, min_child_weight):
        """
        Run cross validation on a range of hyperparameters

        OUTPUT:
        minimum binary log loss based on cross validation result
        """
        params = {
            "objective" : "binary",
            "bagging_fraction" : 0.2,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "reg_alpha": 1,
            "reg_lambda": 1,
            "boosting": "gbdt",
            "subsample" : 0.8,
            "colsample_bytree" : 0.8,
            "verbosity": -1,
            "metric" : 'normalized_entropy'
        }

        params['n_estimators'] = int(round(n_estimators))
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['num_leaves'] = int(round(num_leaves))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight

        cv_result = lgb.cv(params, train_data, feval=normalize_entropy, nfold=n_folds, seed=random_seed \
            , verbose_eval =200,stratified=False)

        return (-1.0 * np.array(cv_result['normalized_entropy-mean'])).max()

    # range
    lgb_bo = BayesianOptimization(lgb_eval, {'n_estimators': (100,300),
                                            'learning_rate': (0.01, 1.0),
                                            'feature_fraction': (0.1, 0.9),
                                            'max_depth': (5, 9),
                                            'num_leaves' : (200,300),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgb_bo.maximize(init_points=init_round, n_iter=opt_round,acq='ei')

    # return best parameters
    return max(lgb_bo.res, key=lambda x:x['target'])['params']

def create_train_gbm_model(training_df, testing_df):
    """
    Train GBDT model

    INPUT:
    training_df: dataframe
    testing_df: dataframe

    OUTPUT:
    x_train: training feature dataframe (to be used in hybrid and benchmark models later)
    x_valid: validation feature dataframe (to be used in hybrid and benchmark models later)
    y_train: training label (to be used in hybrid and benchmark models later)
    y_valid: validation label (to be used in hybrid and benchmark models later)
    model: GBDT model with the optimized hyperparameters obtained from beysian optimizer
    """

    column_excl = ['index', 'person', 'offer_id', 'received_time', 'label', 'training_label']

    feat = list(np.setdiff1d(testing_df.columns, column_excl))

    x_train, x_valid, y_train, y_valid = train_test_split(training_df[feat], training_df.label \
        , test_size=0.3, random_state=0)

    opt_params = bayes_parameter_opt_lgb(x_train, y_train)

    print(opt_params)

    # Create LightGBM model
    gbm = lgb.LGBMRegressor(objective='binary',
                        metric='normalized_entropy',
                        feature_fraction=round(opt_params['feature_fraction'],2),
                        n_estimators=int(round(opt_params['n_estimators'])),
                        learning_rate=round(opt_params['learning_rate'],2),
                        max_depth = int(round(opt_params['max_depth'])),
                        min_child_weight=int(round(opt_params['min_child_weight'])),
                        min_split_gain=round(opt_params['min_split_gain'],3),
                        num_leaves=int(round(opt_params['num_leaves'])))

    # train model
    gbm = lgb.LGBMRegressor(objective='binary',
                        metric='normalized_entropy',
                        bagging_fraction=0.2,
                        bagging_freq=1,
                        min_child_samples=20,
                        reg_alpha=1,
                        reg_lambda=1,
                        boosting="gbdt",
                        subsample=0.8,
                        colsample_bytree=0.8,
                        verbosity=-1,
                        feature_fraction=round(opt_params['feature_fraction'],2),
                        n_estimators=int(round(opt_params['n_estimators'])),
                        learning_rate=round(opt_params['learning_rate'],2),
                        max_depth = int(round(opt_params['max_depth'])),
                        min_child_weight=int(round(opt_params['min_child_weight'])),
                        min_split_gain=round(opt_params['min_split_gain'],3),
                        num_leaves=int(round(opt_params['num_leaves'])))

    model = gbm.booster_

    # Plot GBDT tree-based feature importance
    lgb.plot_importance(model, max_num_features=20, figsize=(10,10))

    return x_train, x_valid, y_train, y_valid, model

def add_tree_based_features(model, dataframe, feat):
    """
    Create tree based features using GBDT trained model
    Combine tree based features with the original features

    INPUT:
    model: GBDT model trained
    df: dataframe to create tree based features on (e.g. training_df, testing_df)
    feat: list of features to create tree based features on

    OUTPUT:
    dataframe with tree based features and the original features
    """

    # Create tree-based features with trained GBDT
    gbdt_feats = model.predict(dataframe[feat], pred_leaf=True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats.shape[1])]

    # Perform one-hot encoding on features created by GBDT
    gbdt_feats_df = pd.DataFrame(gbdt_feats, columns = gbdt_feats_name)
    ohe_df = pd.get_dummies(gbdt_feats_df)

    # Combine original features and one-hot encoding tree-based features
    return pd.concat([dataframe[feat].reset_index(), ohe_df], axis = 1), dataframe.label

def train_logistic_regression_tree_features(model, training_df, testing_df):
    """
    train logistic regression on features space including both original and tree based features

    INPUT:
    model: logistic regression
    training_df: dataframe
    testing_df: dataframe

    OUTPUT:
    normalized entropy for training/validation/testing data
    predicted label on testing data to be saved in an output file
    """

    column_excl = ['index', 'person', 'offer_id', 'received_time', 'label', 'training_label']
    feat = list(np.setdiff1d(testing_df.columns, column_excl))

    #train logistic regression with tree features
    x_train_hybrid, y_train_hybrid = add_tree_based_features(model,training_df, feat)
    x_test_hybrid, y_test_hybrid = add_tree_based_features(model,testing_df, feat)

    # split training data into training and validation
    x_train_hybrid, x_valid_hybrid, y_train_hybrid, y_valid_hybrid = \
    train_test_split(x_train_hybrid, y_train_hybrid, test_size = 0.3, random_state = 0)

    # train logistic regression model
    lr_tree = LogisticRegressionCV(cv=5, random_state=0, scoring='neg_log_loss')\
    .fit(x_train_hybrid, y_train_hybrid)

    # return normalized entropy for training data
    train_ne_hybrid = return_normalize_entropy(lr_tree.predict_proba(x_train_hybrid)[:,1], \
        y_train_hybrid.values)

    # return normalized entropy for validation data
    val_ne_hybrid = return_normalize_entropy(lr_tree.predict_proba(x_valid_hybrid)[:,1], \
        y_valid_hybrid.values)

    # return normalized entropy for testing data
    test_ne_hybrid = return_normalize_entropy(lr_tree.predict_proba(x_test_hybrid)[:,1], \
        y_test_hybrid.values)

    # turn predicted probability result to 1 or 0
    pred = pd.DataFrame(pd.Series([1 if x >0.5 else 0 for x in \
        lr_tree.predict_proba(x_test_hybrid)[:,1]])).reset_index()

    return train_ne_hybrid, val_ne_hybrid, test_ne_hybrid \
    , pd.concat([x_test_hybrid.reset_index(), pred], axis=1)

def train_logistic_regression(x_train, x_valid, y_train, y_valid, testing_df):
    """
    train logistic regression on original features only

    INPUT:
    x_train: training features
    x_valid: validation features
    y_train: training labels
    y_valid: validation labels
    testing_df: dataframe

    OUTPUT:
    normalized entropy for training/validation/testing data
    predicted label on testing data to be saved in an output file
    """

    feat = x_train.columns
    x_test = testing_df[feat]

    logistic_regression = LogisticRegressionCV(cv=5, random_state=0, scoring='neg_log_loss') \
    .fit(x_train, y_train)

    # return normalized entropy for training data
    train_ne_lr = return_normalize_entropy(logistic_regression.predict_proba(x_train)[:, 1]\
        , y_train.values)

    # return normalized entropy for validation data
    val_ne_lr = return_normalize_entropy(logistic_regression.predict_proba(x_valid)[:, 1]\
        , y_valid.values)

    # return normalized entropy for testing data
    test_ne_lr = return_normalize_entropy(logistic_regression.predict_proba(x_test)[:, 1]\
        ,testing_df.label.values)

    # turn predicted probability result to 1 or 0
    pred = pd.DataFrame(pd.Series([1 if x >0.5 else 0 for x in \
        logistic_regression.predict_proba(x_test)[:,1]])).reset_index()

    return train_ne_lr, val_ne_lr, test_ne_lr, pd.concat([x_test.reset_index(), pred], axis=1)

def train_lgbm(model, x_train, x_valid, y_train, y_valid, testing_df):
    """
    train LightGBM on original features only

    INPUT:
    model: the same GBDT model trained previously to create tree features
    x_train: training features
    x_valid: validation features
    y_train: training labels
    y_valid: validation labels
    testing_df: dataframe

    OUTPUT:
    normalized entropy for training/validation/testing data
    predicted label on testing data to be saved in an output file
    """

    feat = x_train.columns
    x_test = testing_df[feat]

    # return normalized entropy for training data
    train_ne_gbm = return_normalize_entropy(model.predict(x_train), y_train.values)

    # return normalized entropy for validation data
    val_ne_gbm = return_normalize_entropy(model.predict(x_valid), y_valid.values)

    # return normalized entropy for testing data
    test_ne_gbm = return_normalize_entropy(model.predict(x_test), testing_df.label.values)

    # turn predicted probability result to 1 or 0
    pred = pd.DataFrame(pd.Series([1 if x >0.5 else 0 for x in \
        model.predict(x_test)])).reset_index()

    return train_ne_gbm, val_ne_gbm, test_ne_gbm, pd.concat([x_test.reset_index(), pred], axis=1)
