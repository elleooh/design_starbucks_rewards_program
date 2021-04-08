import pandas as pd
import utils as u

def transform_portfolio(portfolio):
    """
    The original format for portfolio's 'channel' column is [web, email, mobile, social]
    This function makes each channel as a separate column. 
    1 or 0 to indicate whether the portfolio is of this channel

    INPUT:
    portfolio: dataframe

    OUTPUT:
    portfolio: dataframe
    """

    # email is not included because all portfolios use email as a channel
    channels_lst = ['web', 'mobile', 'social']

    for channel in channels_lst:
        portfolio[channel] = portfolio['channels'].apply(lambda x: 1 if channel in x else 0)

    portfolio.drop(columns=['channels'], inplace=True)

    portfolio = portfolio.set_index('id')

    # save dataframe to database 
    u.save_dataframe_to_sql(portfolio, 'portfolio')

    return portfolio

def transform_profile(profile):
    """
    Transform profile features - became_member_on, gender, and age

    INPUT:
    profile: dataframe

    OUTPUT:
    profile: dataframe
    """

    # turn 'became_member_on' to datetime
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')

    # turn gender == 'o' to Unknown
    profile.loc[profile['gender'] == 'O', 'gender'] = None

    # turn age > 100 to Unknown
    profile.loc[profile['age'] > 100, 'age'] = None

    profile = profile.set_index('id')

    # save dataframe to database
    u.save_dataframe_to_sql(profile, 'profile')

    return profile

def transform_transcript(transcript):
    """
    The original format for transcript's 'value' column is
    {'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9', 'amount:' 10}

    INPUT:
    transcript: dataframe

    OUTPUT:
    transcript: dataframe
    """

    # add 'offer id' and 'amount' columns to the dataframe
    col_names = ['offer id', 'amount']

    for col in col_names:
        col_lst = [value[col] if col in value else None for value in transcript['value']]
        transcript[col] = col_lst

    # turn amount > 50 to None
    transcript.loc[transcript['amount'] > 50, 'amount'] = None

    transcript.drop(columns=['value'], inplace=True)
    transcript.rename(columns={'offer id': 'offer_id'}, inplace=True)

    # save dataframe to database 
    u.save_dataframe_to_sql(transcript, 'transcript')

    return transcript
