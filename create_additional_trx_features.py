import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

def create_final_transcript(transcript, sql_connect):
	# If offer was received and viewed, label = 1; otherwise 0
	query = """
		SELECT DISTINCT a.person, a.offer_id, a.time as received_time, b.time as viewed_time
		, CASE WHEN b.person IS NULL THEN 0 ELSE 1 END as label
		FROM (SELECT person, offer_id, time 
		FROM transcript
		WHERE event = 'offer received') a
		LEFT JOIN (SELECT person, offer_id, time
		FROM transcript
		WHERE event = 'offer viewed') b
		    ON a.person = b.person
		    AND a.offer_id = b.offer_id
		    AND a.time <= b.time
		"""

	pd.read_sql_query(query,sql_connect).to_sql('transcript_rec_view', sql_connect, if_exists='replace')

	# For customers receiving and viewing the same offers, use the min view time as the only view time for that offer
	query = """
		SELECT a.person, a.offer_id, a.received_time, a.viewed_time-a.received_time as duration_view, label
		FROM (SELECT person, offer_id, received_time, min(viewed_time) as viewed_time, max(label) as label
		FROM transcript_rec_view
		GROUP BY person, offer_id, received_time) a
		"""
	
	transcript_final = pd.read_sql_query(query,sql_connect)

	assert sum(transcript['event'] == 'offer received') == transcript_final.shape[0]\
	, "Incorrect dimension - the number of 'offer received' in original transcript should equal to transcript_final"

	transcript_final.to_sql('transcript_final', sql_connect, if_exists='replace')

	return transcript_final

def splt_training_testing(transcript_final, sql_connect):
	# To prevent data leakage when creating features based on transcript
	# if the time is smaller than a threshold, make it training data and larger than a threshold as testing data
	# training : testing should roughly be 3:1

	transcript_quantile = transcript_final.groupby('person')['received_time'].quantile(0.75).reset_index()
	transcript_quantile.to_sql('transcript_quantile', sql_connect, if_exists='replace')

	query = """
		SELECT a.*, CASE WHEN a.received_time <= b.received_time THEN 1 ELSE 0 END as training_label
		FROM transcript_final a
		LEFT JOIN transcript_quantile b
		        ON a.person = b.person    
		"""

	assert pd.read_sql_query(query,sql_connect).shape[0]==transcript_final.shape[0] \
	, "Wrong data dimension after joining with quantile"

	transcript_final = pd.read_sql_query(query,sql_connect)

	assert transcript_final['training_label'].sum()/ transcript_final.shape[0] > 0.7 \
	, "Training data accounts for less than 70% of the total data"

	transcript_training = transcript_final[transcript_final['training_label']==1]
	transcript_testing = transcript_final[transcript_final['training_label']==0]

	assert len(transcript_training['person'].unique()) >= len(transcript_testing['person'].unique()) \
	, "Training data include fewer customer than in the testin data"

	transcript_final.to_sql('transcript_final', sql_connect, if_exists='replace')
	transcript_training.to_sql('transcript_training', sql_connect, if_exists='replace')
	transcript_testing.to_sql('transcript_testing', sql_connect, if_exists='replace')

	return transcript_training, transcript_testing

def create_features_using_groupby(transcript_training, entity_col, feature, avg=True, min=True, max=True):
    groupby = transcript_training.groupby(entity_col)[feature]
    
    features, col_name = [], []
    if avg:
        features.append(groupby.mean())
        col_name.append('avg_'+feature)
    if min:
        features.append(groupby.min())
        col_name.append('min_'+feature)
    if max:
        features.append(groupby.max())
        col_name.append('max_'+feature)
        
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = col_name
    
    return feature_df

def create_features_offer(portfolio, transcript_training, sql_connect):
	portfolio_duration = create_features_using_groupby(transcript_training, 'offer_id', 'duration_view')
	portfolio_view_rate = create_features_using_groupby(transcript_training, 'offer_id', 'label', min=False, max=False)
	portfolio_view_rate.columns=['view_rate']
	portfolio_feat = pd.concat([portfolio_view_rate, portfolio_duration], axis=1)
	assert portfolio_feat.shape[0] == portfolio.shape[0], "rows do not match with original data (portfolio)"
	portfolio = portfolio.join(portfolio_feat)

	# remove constant and highly correlated features
	portfolio.drop(columns=['min_duration_view', 'difficulty', 'mobile', 'view_rate', 'avg_duration_view'], inplace=True)

	portfolio.to_sql('portfolio', sql_connect, if_exists='replace')

	return portfolio

def create_features_customer(profile, transcript_training, sql_connect):
	query = """
		SELECT a.person, min(amount) as min_amount, max(amount) as max_amount, avg(amount) as avg_amount
		FROM transcript a
		    JOIN transcript_quantile b
		        ON a.person = b.person  
		WHERE a.time <= b.received_time
		GROUP BY a.person
		"""

	profile_amount = pd.read_sql_query(query,sql_connect).set_index('person')

	profile_duration = create_features_using_groupby(transcript_training, 'person', 'duration_view')

	profile_view_rate = create_features_using_groupby(transcript_training, 'person', 'label', min=False, max=False)
	profile_view_rate.columns=['view_rate']

	profile_trx_rate = (transcript_training.groupby('person').size()*100/(transcript_training.groupby('person')['received_time'].max() - transcript_training.groupby('person')['received_time'].min())).reset_index()
	profile_trx_rate.columns = ['person', 'avg_trx_cnt']
	profile_trx_rate.loc[profile_trx_rate['avg_trx_cnt']==np.inf, 'avg_trx_cnt'] = 1
	profile_trx_rate = profile_trx_rate.set_index('person')

	profile_feat = profile_amount.join(profile_duration).join(profile_view_rate).join(profile_trx_rate)

	assert pd.merge(profile, profile_feat, how='left', left_index=True, right_index=True).shape[0] == profile.shape[0], "rows do not match with original data (profile)"

	profile = pd.merge(profile, profile_feat, how='left', left_index=True, right_index=True)

	# transform skewed data using log transformation
	view_amount_features = ['max_duration_view', 'view_rate', 'max_amount', 'min_duration_view', 'min_amount',\
                       'avg_amount', 'avg_trx_cnt', 'avg_duration_view']
	profile_transformed = np.log(profile[view_amount_features]+1)

	profile = pd.concat([profile[['gender', 'age', 'became_member_on', 'income']],profile_transformed], axis=1)

	profile.drop(columns=['income', 'min_amount', 'avg_amount', 'avg_duration_view'], inplace=True)

	profile.to_sql('profile', sql_connect, if_exists='replace')

	return profile

def log_transform_transcript(transcript_training, transcript_testing):
	transcript_training['duration_view'] = np.log(transcript_training['duration_view']+1)
	transcript_testing['duration_view'] = np.log(transcript_testing['duration_view']+1)
	return transcript_training, transcript_testing

def combine_portfolio_profile_transcript(transcript_training, transcript_testing, portfolio, profile):
	# combine training and testing
	transcript_comb = pd.concat([transcript_training, transcript_testing], axis=0)

	trans_port_df = pd.merge(transcript_comb, portfolio, how='inner', left_on='offer_id', right_on='id')
	trans_port_df.drop(columns=['index'], inplace=True)
	trans_port_df.rename({'max_duration_view':'max_duration_view_portfolio'}, axis=1, inplace=True)
	assert trans_port_df.shape[0] == transcript_comb.shape[0], "rows do not match with original data"

	trans_port_profile_df = pd.merge(trans_port_df, profile, how='inner', left_on='person', right_on='id')
	#trans_port_profile_df.drop(columns=['id'], inplace=True)
	trans_port_profile_df.rename({'max_duration_view':'max_duration_view_profile'}, axis=1, inplace=True)
	assert trans_port_profile_df.shape[0] == transcript_comb.shape[0], "rows do not match with original data"

	missing_col = list(trans_port_profile_df.isnull().sum()[trans_port_profile_df.isnull().sum() > 0].index)

	for col in missing_col:
		if trans_port_profile_df[col].dtypes == 'object':
			# Using mode to impute the missing categorical values
			trans_port_profile_df.loc[trans_port_profile_df.loc[:,col].isnull(),col]= \
			trans_port_profile_df[col].value_counts().sort_values(ascending=False).index[0]
		else:
			# Using mean to impute the missing numerical values
			trans_port_profile_df.loc[trans_port_profile_df.loc[:,col].isnull(),col]=trans_port_profile_df.loc[:,col].mean()

	return trans_port_profile_df

def transform_feature_combine_df(combine_df):
	# Change became_member_on to date difference between min became_member_on
	min_became_member_on = pd.to_datetime(combine_df['became_member_on']).min()
	combine_df['became_member_on'] = (pd.to_datetime(combine_df['became_member_on']) - min_became_member_on).astype('timedelta64[D]')

	#OHE for offer_type and gender
	combine_df = pd.concat([combine_df, pd.get_dummies(combine_df[['offer_type', 'gender']], drop_first=True)], axis=1)

	combine_df.drop(columns=['offer_type', 'gender'], inplace=True)

	return combine_df
