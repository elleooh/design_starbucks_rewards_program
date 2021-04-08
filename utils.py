import sqlite3
import pandas as pd

def save_dataframe_to_sql(data_frame, table_name):
    """
    Save dataframe specified as SQL table (name provided as tableNmae) in Starbucks.db

    INPUT:
    data_frame: DataFrame
    table_name: String

    OUTPUT:
    None
    """
    sql_connect = sqlite3.connect('starbucks.db')
    data_frame.to_sql(table_name, sql_connect, if_exists='replace')

def read_dataframe_from_sql(query):
    """
    Read table from Starbucks.db to dataframe

    INPUT:
    query: String

    OUTPUT:
    DataFrame
    """
    sql_connect = sqlite3.connect('starbucks.db')
    return pd.read_sql_query(query,sql_connect)
