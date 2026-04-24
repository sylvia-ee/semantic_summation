import pandas as pd
from collections import Counter
import numpy as np 

def childes_cleaner(childes_df, language_column="gloss", identifier_col="id"):

    """
    1st-pass tokenization of childes_df transcripts "gloss" on spaces only.
    Returns dict of each transcript id and a list of tokens of each word in transcript

    :param childes_df: pd df containing the data
    :param language_column: str for column containing text to tokenize
    :param identifier_col: str for column containing unique identifiers
    :return: dict {id: [token1, token2, ...]} of each transcript and the tokens inside
    """

    df = childes_df[[identifier_col, language_column]].copy()
    tokens_series = df[language_column].fillna("").astype(str).str.split()
    id_token_dict = dict(zip(df[identifier_col], tokens_series))

    return id_token_dict



def count_words_in_childes(childes_df, language_column="gloss"):
    """
    Count occurrences of each word token across the CHILDES corpus.
    Splits on whitespace only (1st-pass tokenization), lowercases all tokens.

    :param childes_df: pd df containing the CHILDES data
    :param language_column: str for column containing text to tokenize (default "stem")
    :return: Counter mapping lowercased word -> count
    """
    tokens = (
        childes_df[language_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.split()
        .explode()
    )
    return Counter(tokens.dropna())
