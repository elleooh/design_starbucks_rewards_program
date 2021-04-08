import pandas as pd

def load_portfolio(portfolio_filepath):
    """
    Load Portfolio json file to dataframe

    INPUT:
    portfolio_filepath: string

    OUTPUT:
    portfolio: dataframe
    """
    portfolio = pd.read_json(portfolio_filepath, orient='records', lines=True)
    return portfolio

def load_profile(profile_filepath):
    """
    Load Profile json file to dataframe

    INPUT:
    profile_filepath: string

    OUTPUT:
    profile: dataframe
    """
    profile = pd.read_json(profile_filepath, orient='records', lines=True)
    return profile

def load_transcript(transcript_filepath):
    """
    Load Transcript json file to dataframe

    INPUT:
    transcript_filepath: string

    OUTPUT:
    transcript: dataframe
    """
    transcript = pd.read_json(transcript_filepath, orient='records', lines=True)
    return transcript
