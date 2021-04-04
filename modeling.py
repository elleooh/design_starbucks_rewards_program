import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss, classification_report
#%matplotlib inline


# Hybrid GBDT (LightGBM) & LR model
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=0, n_estimators=1000, learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    
    # parameters
    def lgb_eval(learning_rate, num_leaves, feature_fraction, max_depth , min_split_gain, min_child_weight):
        params = {
            "objective" : "regression", 
            "bagging_fraction" : 0.2, 
            "bagging_freq": 1,
            "min_child_samples": 20, 
            "reg_alpha": 1, 
            "reg_lambda": 1,
            "boosting": "gbdt",
            "learning_rate" : 0.01, 
            "subsample" : 0.8, 
            "colsample_bytree" : 0.8, 
            "verbosity": -1, 
            "metric" : 'binary_logloss'
        }
        
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['num_leaves'] = int(round(num_leaves))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, verbose_eval =200,stratified=False)
        
        return (-1.0 * np.array(cv_result['binary_logloss-mean'])).max()
    
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 1.0),
                                            'feature_fraction': (0.1, 0.9),
                                            'max_depth': (5, 9),
                                            'num_leaves' : (200,300),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round,acq='ei')

    # output optimization process
    if output_process: 
        lgbBO.points_to_csv("bayes_opt_result.csv")

    # return best parameters
    return max(lgbBO.res, key=lambda x:x['target'])['params']


def create_train_gbm_model(training_df, testing_df, feat):
	X_train, X_valid, y_train, y_valid = train_test_split(training_df[feat], training_df.label, test_size=0.3, random_state=2021, stratify = training_df.label)

	opt_params = bayes_parameter_opt_lgb(X_train, y_train, init_round=10, opt_round=10, n_folds=5, random_seed=0, n_estimators=1000, learning_rate=0.01)

	print(opt_params)
	
	# Create LightGBM model
	gbm = lgb.LGBMRegressor(objective='binary',
                        feature_fraction=round(opt_params['feature_fraction'],2),
                        learning_rate=round(opt_params['learning_rate'],2),
                        max_depth = int(round(opt_params['max_depth'])),
                        min_child_weight=int(round(opt_params['min_child_weight'])),
                        min_split_gain=round(opt_params['min_split_gain'],3),
                        num_leaves=int(round(opt_params['num_leaves'])))
	
	# train model
	gbm.fit(X_train, y_train,
        eval_set = [(X_train, y_train), (X_valid, y_valid)],
        eval_names = ['train', 'val'],
        eval_metric = 'binary_logloss',
        )

	return X_train, X_valid, y_train, y_valid, gbm

def plot_feature_importance(model):
	lgb.plot_importance(model, max_num_features=20, figsize=(10,10))

def create_tree_based_features(model, df, feat):
    # Create tree-based features with trained GBDT
    gbdt_feats = model.predict(df[feat],pred_leaf=True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats.shape[1])]
    
    # Perform one-hot encoding on features created by GBDT
    gbdt_feats_df = pd.DataFrame(gbdt_feats, columns = gbdt_feats_name) 
    ohe_df = pd.get_dummies(gbdt_feats_df)
    
    # Combine original features and one-hot encoding tree-based features
    return pd.concat([df[feat].reset_index(), ohe_df], axis = 1), df.label

def return_normalize_entropy(predicted, target):
	if len(predicted) != len(target):
		return 'lengths not equal!'

	N = len(target)
	p = np.mean(target)

	numerator = -(1/N)*sum([target[i]*np.log(predicted[i]) + 
		(1-target[i])*np.log(1-predicted[i]) for i in range(N)])

	denominator = -(p*np.log(p)+(1-p)*np.log(1-p))

	return numerator/denominator


def train_logistic_regression_tree_features(model, training_df, testing_df, feat):
	#train logistic regression with tree features
	X_train_hybrid, y_train_hybrid = create_tree_based_features(model,training_df, feat)
	X_test_hybrid, y_test_hybrid = create_tree_based_features(model,testing_df, feat)
	
	X_train_hybrid, X_valid_hybrid, y_train_hybrid, y_valid_hybrid = train_test_split(X_train_hybrid, y_train_hybrid, 
		test_size = 0.3, random_state = 2021)

	lr_tree = LogisticRegressionCV(cv=5, random_state=2021, scoring='neg_log_loss').fit(X_train_hybrid, y_train_hybrid)
	
	train_NE_hybrid = return_normalize_entropy(lr_tree.predict_proba(X_train_hybrid)[:,1], y_train_hybrid.values)

	val_NE_hybrid = return_normalize_entropy(lr_tree.predict_proba(X_valid_hybrid)[:,1], y_valid_hybrid.values)

	test_NE_hybrid = return_normalize_entropy(lr_tree.predict_proba(X_test_hybrid)[:,1], y_test_hybrid.values)

	pred = pd.DataFrame(pd.Series([1 if x >0.5 else 0 for x in lr_tree.predict_proba(X_test_hybrid)[:,1]])).reset_index()

	return train_NE_hybrid, val_NE_hybrid, test_NE_hybrid, pd.concat([X_test_hybrid.reset_index(), pred], axis=1)

def train_logistic_regression(X_train, X_valid, y_train, y_valid, testing_df, feat):
	X_test = testing_df[feat]

	lr = LogisticRegressionCV(cv=5, random_state=2021, scoring='neg_log_loss').fit(X_train, y_train)
	
	# print out result for LR-only model
	train_NE_lr = return_normalize_entropy(lr.predict_proba(X_train)[:, 1], y_train.values)

	val_NE_lr = return_normalize_entropy(lr.predict_proba(X_valid)[:, 1], y_valid.values)

	test_NE_lr = return_normalize_entropy(lr.predict_proba(X_test)[:, 1], testing_df.label.values)

	pred = pd.DataFrame(pd.Series([1 if x >0.5 else 0 for x in lr.predict_proba(X_test)[:,1]])).reset_index()

	return train_NE_lr, val_NE_lr, test_NE_lr, pd.concat([X_test.reset_index(), pred], axis=1)

def train_lgbm(model, X_train, X_valid, y_train, y_valid, testing_df, feat):
	X_test = testing_df[feat]

	train_NE_gbm = return_normalize_entropy(model.predict(X_train), y_train.values)

	val_NE_gbm = return_normalize_entropy(model.predict(X_valid), y_valid.values)

	test_NE_gbm = return_normalize_entropy(model.predict(X_test), testing_df.label.values)

	pred = pd.DataFrame(pd.Series([1 if x >0.5 else 0 for x in model.predict(X_test)])).reset_index()

	return train_NE_gbm, val_NE_gbm, test_NE_gbm, pd.concat([X_test.reset_index(), pred], axis=1)


