'''
Module to ingest & structure Textract table output including:
- Identify numeric, alpha and date columns
- Identify header
- Bring everything together into a standardized table
'''
import logging
import os

import numpy as np
import pandas as pd
from dateutil.parser import parse
# Function to convert numbers encapsulated by () to negative, and remove $
def numeric_cleaner(df_col):
    return (pd.to_numeric(df_col.replace('[\$,)]', '', regex=True).replace( '[(]', '-', regex=True ), 
                          errors='coerce'))

# identify columns that are actually numeric, with parameter for threshold of % nan
def numeric_selector(orig_table, num_table, nan_max_share=0.5):
    # Calculate share of na rows per column
    na_share = num_table.isnull().sum() / len(num_table)
    na_share = na_share[na_share < 0.75]
    
    edited_table = orig_table.copy()
    for col in na_share.index:
        edited_table[[col]] = num_table[[col]]
    
    return edited_table

def is_alpha(value):
    """
    Return whether a value contains alpha characters
    
    :param value: object of any type, value to check if contains characters
    """
    if type(value) != str:
        return False
    else:
        return any(c.isalpha() for c in str(value))
    
    
# Detect if column text can be represented as a date
# Add flag for the dataset that indicates the date type 
def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(str(string), fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def extra_row_remover(df):
    #Drop rows that have only one filled-in value
    df['na_count'] = df.isnull().sum(axis=1)
    df = df[df.na_count < (len(df.columns)-2)]
    df = df.drop('na_count', axis=1).reset_index(drop=True)
    return df