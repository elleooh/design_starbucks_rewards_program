import pandas as pd
import numpy as np
import utils as u

def create_final_transcript():
    """
    Create final transcript with target lable
    indicating whether an offer was received and viewed or not viewed

    OUTPUT:
    transcript_final: dataframe
    """

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
    transcript_rec_view = u.read_dataframe_from_sql(query)

    u.save_dataframe_to_sql(transcript_rec_view, 'transcript_rec_view')

    # For customers receiving and viewing the same offers,
    # use the min view time as the only view time for that offer
    query = """
        SELECT a.person, a.offer_id, a.received_time, a.viewed_time-a.received_time as duration_view, label
        FROM (SELECT person, offer_id, received_time, min(viewed_time) as viewed_time, max(label) as label
            FROM transcript_rec_view
        GROUP BY person, offer_id, received_time) a
        """

    transcript_final = u.read_dataframe_from_sql(query)

    u.save_dataframe_to_sql(transcript_final, 'transcript_final')

    return transcript_final

def split_training_testing():
    """
    Split transcript_final to training and testing based on received_time.
    The training data contain historical data till a received_time.
    The testing data should contain all data after the received_time.

    OUTPUT:
    transcript_training: dataframe
    transcript_testing: dataframe
    """

    transcript_final = create_final_transcript()

    # To prevent data leakage when creating features based on transcript
    # if the time is smaller than a threshold, make it training data
    # and larger than a threshold as testing data
    # training : testing should roughly be 3:1
    transcript_quantile = transcript_final.groupby('person')['received_time']\
    .quantile(0.75).reset_index()

    query = """
        SELECT a.*, CASE WHEN a.received_time <= b.received_time THEN 1 ELSE 0 END as training_label
        FROM transcript_final a
        LEFT JOIN transcript_quantile b
                ON a.person = b.person    
        """

    assert u.read_dataframe_from_sql(query).shape[0]==transcript_final.shape[0] \
    , "Wrong data dimension after joining with quantile"

    transcript_final = u.read_dataframe_from_sql(query)

    assert transcript_final['training_label'].sum()/ transcript_final.shape[0] > 0.7 \
    , "Training data accounts for less than 70% of the total data"

    # split transcript_final to training and testing
    transcript_training = transcript_final[transcript_final['training_label']==1]
    transcript_testing = transcript_final[transcript_final['training_label']==0]

    assert len(transcript_training['person'].unique()) >= len(transcript_testing['person'].unique()) \
    , "Training data include fewer customer than in the testing data"

    u.save_dataframe_to_sql(transcript_final, 'transcript_final')
    u.save_dataframe_to_sql(transcript_quantile, 'transcript_quantile')
    u.save_dataframe_to_sql(transcript_training, 'transcript_training')
    u.save_dataframe_to_sql(transcript_testing, 'transcript_testing')

    return transcript_training, transcript_testing

def create_features_using_groupby(training, entity, feature, avg=True, minimum=True, maximum=True):
    """
    Helper function to create portfolio and profile features based on transcript data

    INPUT:
    training: dataframe
    entity: String (i.e. portfolio, profile)
    feature: String (e.g. duratoin view, amount)
    avg: Boolean - if True, create average for the feature provided; if False, do not create
    minimum: Boolean - if True, create minimum for the feature provided; if False, do not create
    maximum: Boolean - if True, create maxinum for the feature provided; if False, do not create

    OUTPUT:
    feature_df: dataframe
    """

    entity_col = 'offer_id' if entity == 'portfolio' else 'person'

    groupby = training.groupby(entity_col)[feature]

    features, col_name = [], []
    if avg:
        features.append(groupby.mean())
        col_name.append('avg_'+feature)
    if minimum:
        features.append(groupby.min())
        col_name.append('min_'+feature)
    if maximum:
        features.append(groupby.max())
        col_name.append('max_'+feature)

    feature_df = pd.concat(features, axis=1)
    feature_df.columns = [col + '_' + entity for col in col_name]

    return feature_df

def create_features_offer(portfolio, transcript_training):
    """
    Create offer (portfolio) features based on transcript data
    Use training data only to prevent data leakage

    INPUT:
    portfolio: dataframe
    stranscript_training: dataframe

    OUTPUT:
    portfolio: dataframe
    """

    # create avg/min/max duration view
    portfolio_duration = create_features_using_groupby(transcript_training, \
    	'portfolio', 'duration_view')

    # create view rate (average of label)
    portfolio_view_rate = create_features_using_groupby(transcript_training, \
    	'portfolio', 'label', minimum=False, maximum=False)
    portfolio_view_rate.columns=['view_rate_portfolio']

    portfolio_feat = pd.concat([portfolio_view_rate, portfolio_duration], axis=1)

    assert portfolio_feat.shape[0] == portfolio.shape[0], \
    "rows do not match with original data (portfolio)"

    portfolio = portfolio.join(portfolio_feat)

    # remove constant and highly correlated features
    portfolio.drop(columns=['min_duration_view_portfolio', 'difficulty', \
    	'mobile', 'view_rate_portfolio', 'avg_duration_view_portfolio'], inplace=True)

    u.save_dataframe_to_sql(portfolio, 'portfolio')

    return portfolio

def create_features_customer(profile, transcript_training):
    """
    Create customer (profile) features based on transcript data
    Use training data only to prevent data leakage

    INPUT:
    profile: dataframe
    transcript_training: dataframe

    OUTPUT:
    profile: dataframe
    """

    # create avg/min/max amount features. Need to calculate amount features from transcript
    # because transcript_training only contains transactions for offer received and viewed.
    # such transactions do not have amount associated

    query = """
        SELECT a.person, min(amount) as min_amount, max(amount) as max_amount, avg(amount) as avg_amount
        FROM transcript a
            JOIN transcript_quantile b
                ON a.person = b.person  
        WHERE a.time <= b.received_time
        GROUP BY a.person
        """

    profile_amount = u.read_dataframe_from_sql(query).set_index('person')

    # create avg/min/max amount duration_view
    profile_duration = create_features_using_groupby(transcript_training\
    	, 'profile', 'duration_view')

    # create view rate (average of label)
    profile_view_rate = create_features_using_groupby(transcript_training, 'profile', 'label'\
    	, minimum=False, maximum=False)
    profile_view_rate.columns=['view_rate_profile']

    # create trx rate (count of transactions per person/(max received time - min received time))
    profile_trx_rate = (transcript_training.groupby('person').size()*100\
        /(transcript_training.groupby('person')['received_time'].max() \
            - transcript_training.groupby('person')['received_time'].min())).reset_index()
    profile_trx_rate.columns = ['person', 'avg_trx_cnt']
    # set trx rate = 1 if max received time == min received time
    profile_trx_rate.loc[profile_trx_rate['avg_trx_cnt']==np.inf, 'avg_trx_cnt'] = 1
    profile_trx_rate = profile_trx_rate.set_index('person')

    profile_feat = profile_amount.join(profile_duration)\
    .join(profile_view_rate).join(profile_trx_rate)

    assert pd.merge(profile, profile_feat, how='left', left_index=True, right_index=True).shape[0] == profile.shape[0]\
    , "rows do not match with original data (profile)"

    profile = pd.merge(profile, profile_feat, how='left', left_index=True, right_index=True)

    return profile

def log_transform_features_customer(profile):
    """
    log transform skewed customer features

    INPUT:
    profile: dataframe

    OUTPUT:
    profile: dataframe
    """

    view_amount_features = ['max_duration_view_profile', 'view_rate_profile', 'max_amount', \
    'min_duration_view_profile', 'min_amount',\
    'avg_amount', 'avg_trx_cnt', 'avg_duration_view_profile']

    profile_transformed = np.log(profile[view_amount_features]+1)

    profile = pd.concat([profile[['gender', 'age', 'became_member_on', 'income']]\
    	,profile_transformed], axis=1)

    profile.drop(columns=['income', 'min_amount', 'avg_amount', 'avg_duration_view_profile']\
    	, inplace=True)

    u.save_dataframe_to_sql(profile, 'profile')

    return profile

def transform_transcript():
    """
    log transform skewed duration view in both training and testing data

    OUTPUT:
    transcript_training: dataframe
    transcript_testing: dataframe
    """

    transcript_training, transcript_testing = split_training_testing()

    # separately transform training and testing to prevent data leakage
    transcript_training['duration_view'] = np.log(transcript_training['duration_view']+1)
    transcript_testing['duration_view'] = np.log(transcript_testing['duration_view']+1)

    return transcript_training, transcript_testing

def combine_portfolio_profile_transcript(training, testing, portfolio, profile):
    """
    combine transcript, portfolio, and profile

    INPUT:
    training: dataframe
    testing: dataframe
    portfolio: dataframe
    profile: dataframe

    OUTPUT:
    combine_df: dataframe
    """

    # combine training and testing
    transcript_comb = pd.concat([training, testing], axis=0)

    trans_port_df = pd.merge(transcript_comb, portfolio, \
    	how='inner', left_on='offer_id', right_on='id')

    trans_port_df.drop(columns=['index'], inplace=True)

    assert trans_port_df.shape[0] == transcript_comb.shape[0]\
    , "rows do not match with original data"

    combine_df = pd.merge(trans_port_df, profile, how='inner', left_on='person', right_on='id')

    assert combine_df.shape[0] == transcript_comb.shape[0], "rows do not match with original data"

    missing_col = list(combine_df.isnull().sum()[combine_df.isnull().sum() > 0].index)

    for col in missing_col:
        if combine_df[col].dtypes == 'object':
            # Using mode to impute the missing categorical values
            combine_df.loc[combine_df.loc[:,col].isnull(),col]= \
            combine_df[col].value_counts().sort_values(ascending=False).index[0]
        else:
            # Using mean to impute the missing numerical values
            combine_df.loc[combine_df.loc[:,col].isnull(),col]=combine_df.loc[:,col].mean()

    return combine_df

def transform_feature_combine_df(combine_df):
    """
    transform features (i.e. 'became_member_on', 'offer_type', 'gender') in combine_df

    INPUT:
    combine_df: dataframe

    OUTPUT:
    combine_df: dataframe
    """

    # Change became_member_on to date difference between min became_member_on
    min_became_member_on = pd.to_datetime(combine_df['became_member_on']).min()
    combine_df['became_member_on'] = (pd.to_datetime(combine_df['became_member_on']) \
    	- min_became_member_on).astype('timedelta64[D]')

    #OHE for offer_type and gender
    combine_df = pd.concat([combine_df, pd.get_dummies(combine_df[['offer_type', 'gender']]\
    	, drop_first=True)], axis=1)

    combine_df.drop(columns=['offer_type', 'gender'], inplace=True)

    return combine_df

def create_combined_df (portfolio, profile):
    """
    put all functions together to create combined dataframe combining
    transcript, portfolio, and profile with clean and transformed features

    INPUT:
    transcript: dataframe
    portfolio: dataframe
    profile: dataframe

    OUTPUT:
    combine_df: dataframe
    """

    transcript_training, transcript_testing = transform_transcript()

    portfolio = create_features_offer(portfolio, transcript_training)

    profile = create_features_customer(profile, transcript_training)

    profile = log_transform_features_customer(profile)

    combine_df = combine_portfolio_profile_transcript(transcript_training, \
    	transcript_testing, portfolio, profile)

    combine_df = transform_feature_combine_df(combine_df)

    return combine_df
