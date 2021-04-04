import pandas as pd
import sqlite3
import json
import matplotlib.pyplot as plt
#%matplotlib inline

def transform_portfolio(portfolio):
	# email is not included because all portfolios use email as a channel
	channels_lst = ['web', 'mobile', 'social']
	channels_df = pd.DataFrame(columns = channels_lst)
	portfolio = pd.concat([portfolio, channels_df], axis=1)

	for channel in channels_lst:
		portfolio[channel] = portfolio['channels'].apply(lambda x: 1 if channel in x else 0)

	portfolio.drop(columns=['channels'], inplace=True)

	return portfolio


def transform_profile(profile):
	profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')

	# Turn gender == 'o' to Unknown
	profile.loc[profile['gender'] == 'O', 'gender'] = None

	# Turn age > 100 to Unknown
	profile.loc[profile['age'] > 100, 'age'] = None

	return profile

def transform_transcript(transcript):
	# Add 'offer id' and 'amount' columns to the dataframe
	col = ['offer id', 'amount']

	for c in col:
		col_lst = []
		for v in transcript['value']:
			try:
				col_lst.append(v[c])
			except:
				col_lst.append(None)
		transcript[c] = col_lst

	transcript.drop(columns=['value'], inplace=True)

	# Turn amount > 50 to Unknown
	transcript.loc[transcript['amount'] > 50, 'amount'] = None

	transcript.rename(columns={'offer id': 'offer_id'}, inplace=True)

	return transcript









