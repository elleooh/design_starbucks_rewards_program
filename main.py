import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_transformation as t
import create_additional_trx_features as f
import modeling as m


if __name__ == '__main__':
	portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
	profile = pd.read_json('data/profile.json', orient='records', lines=True)
	transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

	portfolio = t.transform_portfolio(portfolio)
	profile = t.transform_profile(profile)
	transcript = t.transform_transcript(transcript)

	sql_connect = sqlite3.connect('starbucks.db')
	cursor = sql_connect.cursor()

	portfolio = portfolio.set_index('id')
	profile = profile.set_index('id')

	portfolio.to_sql('portfolio', sql_connect, if_exists='replace')
	profile.to_sql('profile', sql_connect, if_exists='replace')
	transcript.to_sql('transcript', sql_connect, if_exists='replace')

	transcript_final = f.create_final_transcript(transcript, sql_connect)
	transcript_final.to_sql('transcript_final', sql_connect, if_exists='replace')

	transcript_training, transcript_testing = f.splt_training_testing(transcript_final, sql_connect)
	transcript_training, transcript_testing = f.log_transform_transcript(transcript_training, transcript_testing)

	portfolio = f.create_features_offer(portfolio, transcript_training, sql_connect)
	#print(portfolio.isnull().sum())
	profile = f.create_features_customer(profile, transcript_training, sql_connect)
	#print(profile.isnull().sum())
	combine_df = f.combine_portfolio_profile_transcript(transcript_training, transcript_testing, portfolio, profile)
	#print(combine_df.isnull().sum())
	combine_df = f.transform_feature_combine_df(combine_df)
	#print(combine_df.isnull().sum())
	training_df = combine_df[combine_df['training_label'] == 1]
	testing_df = combine_df[combine_df['training_label'] == 0]

	# Hybrid GBDT (LightGBM) & LR model
	## train gbm model for feature transformation
	feat = ['duration_view','reward', 'duration', 'web', 'social',
       'max_duration_view_portfolio', 'age', 'became_member_on',
       'max_duration_view_profile', 'view_rate', 'max_amount',
       'min_duration_view', 'avg_trx_cnt', 'offer_type_discount',
       'offer_type_informational', 'gender_M']

	X_train, X_valid, y_train, y_valid, gbm = m.create_train_gbm_model(training_df, testing_df, feat)
	
	## Plot GBDT tree-based feature importance
	model = gbm.booster_
	# m.plot_feature_importance(model)
	# plt.show()

	# #train logistic regression with  tree features 
	train_NE_hybrid, val_NE_hybrid, test_NE_hybrid, lr_tree_predictions = m.train_logistic_regression_tree_features(model, training_df, testing_df, feat)
	
	#train logistic regression without tree features
	train_NE_lr, val_NE_lr, test_NE_lr, lr_predictions = m.train_logistic_regression(X_train, X_valid, y_train, y_valid, testing_df, feat)

	#train tree model without tree features
	train_NE_gbm, val_NE_gbm, test_NE_gbm, lgbm_predictions = m.train_lgbm(model, X_train, X_valid, y_train, y_valid, testing_df, feat)
	
	print('train-NE (hybrid): ', train_NE_hybrid)
	print('val-NE (hybrid): ', val_NE_hybrid)
	print('test-NE (hybrid): ', test_NE_hybrid)

	print('train-NE (LR-only): ', train_NE_lr)
	print('val-NE (LR-only): ', val_NE_lr)
	print('test-NE (LR-only): ', test_NE_lr)

	print('train-NE (LGBM-only): ', train_NE_gbm)
	print('val-NE (LGBM-only): ', val_NE_gbm)
	print('test-NE (LGBM-only): ', test_NE_gbm)

	# save result to csv
	lr_tree_predictions.to_csv('result/lr_tree_predictions.csv')
	lr_predictions.to_csv('result/lr_predictions.csv')
	lgbm_predictions.to_csv('result/lgbm_predictions.csv')

	cursor.close()











