
# this .py houses all functions related to preprocessing mcdi item-by-item data
# assuming df has at minimum the columns uni_lemma and category

import pandas as pd
import re
import warnings
from collections import defaultdict
import inflect

grammar_machine = inflect.engine()


# ------------- 
# FIRST PASS
# ------------- 

def manual_exclusions(mcdi_ibi_df, word_col="english_gloss", **kwargs):

    """ removes rows from mcdi item-by-item dataframe via explicit user-defined criteria 
    :param mcdi_ibi_df: pd dataframe of mcdi item-by-item data
    :param word_col: str of df column name containing the word
    :param **kwargs: not used here, for call in "merge_mcdi_dict_into_mcdi_df" function
    :return: pd datframe of mcdi item-by-item data with exclusions applied
    """
    # TODO: need to make this more flexible to accept both programmatic rules and .csvs

    # remove proper nouns 
    name_mask = mcdi_ibi_df[word_col].str.endswith(" name")
    mcdi_ibi_df.loc[name_mask, 'excl_reason'] = "ends in ' name'"

    return mcdi_ibi_df

def exclude_nonnouns(
    mcdi_ibi_df,
    nonnoun_cats={
        "games_routines", "action_words", "descriptive_words", "time_words", "locations",
        "descriptive_verbs", "pronouns", "helping_verbs", "sounds",
        "quantifiers", "connecting_words", "NA", "question_words"
    },
    **kwargs
):

    """ removes all rows containing non-nouns using word categories from mcdi item-by-item dataframe 
    :param mcdi_ibi_df: pd dataframe of mcdi item-by-item data
    :param nonnoun_cats: set of all categories denoting non-nouns, to be excluded. default exclusions defined.
    :param **kwargs: not used here (redundant to nonnoun_cats), for call in "merge_mcdi_dict_into_mcdi_df" function
    :return: pd dataframe of mcdi item-by-item data with all rows in excluded categories removed
    """

    nonnoun_cats = set(kwargs.get("nonnoun_cats", nonnoun_cats))

    mask = mcdi_ibi_df['category'].isin(nonnoun_cats)
    mcdi_ibi_df.loc[mask, 'excl_reason'] = "non-noun category"

    return mcdi_ibi_df

def strip_syntax(mcdi_ibi_df, word_col="english_gloss", **kwargs):

    """ reformats special notation and syntax surrounding words to faciliate interpretation of each word as a single token downstream 
    :param mcdi_ibi_df: pd dataframe of mcdi item-by-item data
    :param word_col: str of df column name containing the word to be reformatted
    :param **kwargs: not used here, for call in "merge_mcdi_dict_into_mcdi_df" function
    :return: pd df with each word stripped of special notation and syntax, and additional rows for acceptable "alternative" forms of each word. 
    see additional functions in comments for specific details.
    """

    # search for cases in col uni_lemma and strip using regex.
    # word* -> word
    # word () -> word
    # word ()* -> word
    # word/word -> 2 word alt forms
    # word/word* -> 2 word alt forms
    # word/word () -> 2 word alt forms
    # word/word ()* -> 2 word alt forms

    # store as {original form: [stripped version],
    #           original form: [stripped version, stripped version]}

    # generate df, where each row has an original form in col uni_lemma
    # and each row has one of the stripped versions in col alt_forms
    # e.g. bottom/buttocks returns two rows, 1st row as bottom/buttocks and bottom
    #                                        2nd row as bottom/buttocks and buttocks

    stripped_dict = {}

    for base_wd in mcdi_ibi_df[word_col]:
        cleaned = re.sub(r"\*", "", base_wd).strip()
        alt_no_sense = re.sub(r"\s*\([^)]*\)\s*", "", cleaned).strip()
        alt_forms = [w.strip() for w in alt_no_sense.split('/') if w.strip()]
        stripped_dict[base_wd] = alt_forms

    new_rows = []
    for base_wd, forms in stripped_dict.items():
        for f in forms:
            new_rows.append({word_col: base_wd, 'alt_forms': f})

    alt_form_rows = pd.DataFrame(new_rows)
    mcdi_ibi_df_no_alt = mcdi_ibi_df.drop(columns=['alt_forms'], errors='ignore')
    syntax_cleaned_mcdi_ibi_df = mcdi_ibi_df_no_alt.merge(
        alt_form_rows, on=word_col, how='outer'
    )

    return syntax_cleaned_mcdi_ibi_df

def pp_checker(mcdi_ibi_df_old, mcdi_ibi_df_new, word_col_ibi="english_gloss"):

    """simple comparison to ensure syntax stripping did not result in less words, which should be impossible.
    """

    # makes sure you didn't lose words between dfs
    # should have same set bc exclusion has excl. reason, doesn't remove the row
    unq_wds_old = set(mcdi_ibi_df_old[word_col_ibi].str.lower().unique())
    unq_wds_new = set(mcdi_ibi_df_new[word_col_ibi].str.lower().unique())
    not_shared_wds = unq_wds_old ^ unq_wds_new
    if len(not_shared_wds) > 0:
        warnings.warn(f"{not_shared_wds} are not in both dfs")


def mcdi_cleaner(mcdi_ibi_df, word_col="english_gloss", skip_list=None, func_kwargs=None):

    """ performs all first-pass preprocessing operations for the mcdi item-by-item dataframe.
    :param mcdi_ibi_df: df with at minimum a column for the word and its category
    :param skip_list: specify a list of strs, with strs specifying cleaning operations you don't want to perform.
        "manual_exclusions" skips any manual exclusions you specified.
        "exclude_nonnouns" skips filling in the excl. column using non-noun cats.
        "strip_syntax" skips cleaning word ()*, word (), word*, word/word to just "word".
    :param func_kwargs: dict mapping function name -> dict of kwargs to pass into that function
    :return: df with all original words/data, but extra columns for
        excl_reason: specifying why this row should be excluded
        alt_forms: an alternative form of the word that counts as an instance of that word
            ex. if the base form is "chicken (food)", an accepted alt form is "chicken"
            note that each alternative form gets it own row. so if "chicken (food)" can be
            both "chicken" and "chickens", each one gets its own row.
    """

    # order of funcs in this dict matters b/c performed in order
    # ops_funcs_dict should be all potential preprocessing ops,
    # REGARDLESS of whether they were used.
    ops_funcs_dict = {
        "manual_exclusions": manual_exclusions,
        "exclude_nonnouns": exclude_nonnouns,
        "strip_syntax": strip_syntax
    }

    if skip_list is None:
        skip_list = []

    if func_kwargs is None:
        func_kwargs = {}

    # copy the dataframe to be processed
    mcdi_ibi_df_pp = mcdi_ibi_df.copy() 

    # force all relevant strings into lowercase and drop exact duplicate rows
    mcdi_ibi_df_pp[word_col] = mcdi_ibi_df_pp[word_col].astype(str).str.lower()
    mcdi_ibi_df_pp['category'] = mcdi_ibi_df_pp['category'].astype(str).str.lower()
    mcdi_ibi_df_pp = mcdi_ibi_df_pp.drop_duplicates()

    # initializes empty columns in dataframe
    mcdi_ibi_df_pp['alt_forms'] = None
    mcdi_ibi_df_pp['excl_reason'] = None

    # for each function in the optional functions dictionary,
    # if the user hasn't specified to skip those functions,
    # run them. 
    for name, func in ops_funcs_dict.items():
        if name not in skip_list:
            kwargs = func_kwargs.get(name, {})
            mcdi_ibi_df_pp = func(mcdi_ibi_df_pp, word_col=word_col, **kwargs)

    # run tests to ensure preprocessing was appropriately conducted
    pp_checker(mcdi_ibi_df, mcdi_ibi_df_pp)

    # organize dataframe to sort alphabetically by the primary word key and its alt forms
    mcdi_ibi_pp_df = mcdi_ibi_df_pp.sort_values(by=[word_col, 'alt_forms'])

    return mcdi_ibi_pp_df



# ------------- 
# SECOND PASS
# ------------- 

def create_alt_form_dict(mcdi_ibi_df, main_col='english_gloss', alt_col='alt_forms'):


    """ looks at the mcdi_ibi_df after preprocessing and generates a first-pass dictionary of each lemma and it's alternative forms to facilitate matching of single mcdi word to multiple forms 

    :param mcdi_ibi_df: pd df with at minimum 2 cols, one for the word, another for each alternative form that word has
    :param main_col: str of pd df column name with word to look for
    :param alt_col: str of pd df column name with alternative forms of each word

    :returns alt_map: dictionary of each word in mcdi_ibi_df as {base form: {'alt form', 'alt form', 'alt form'}, ...}
    """

    # initialize dictionary
    alt_map = defaultdict(set)

    # process original df to remove all empty rows and make a copy, then lowercase
    df = mcdi_ibi_df.dropna(subset=[alt_col]).copy()
    df[main_col] = df[main_col].str.lower()
    df[alt_col] = df[alt_col].str.lower()

    # make dictionary of each word and alt forms from df
    for _, row in df.iterrows():
        base = row[main_col]
        alt = row[alt_col]
        alt_map[base].add(alt) # add to dictionary key without overwriting key for base form
    
    alt_map = dict(alt_map) # convert to a regular dictionary from defaultdict

    return alt_map


def manual_inclusions(alt_forms_dict, csv_path, base_col="base", alt_col="alt"):
    """
    Updates existing alt_forms_dict using base/alt pairs from a CSV to include non-programmatically generable alternative forms for words.

    :param alt_forms_dict: existing dictionary {base: set(alt_forms)}
    :param csv_path: path to CSV containing manual inclusions
    :param base_col: column name in CSV for base word
    :param alt_col: column name in CSV for alternate form
    :return: updated dictionary
    """

    d = {k: set(v) for k, v in alt_forms_dict.items()}
    base_keys = set(d.keys())

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        key = row[base_col]
        forms = row[alt_col]

        if key not in base_keys:
            warnings.warn(f"warning: '{key}' not an mcdi word or not the exact mcdi form. check for typos?")

        if pd.isna(key) or pd.isna(forms):
            continue

        if isinstance(forms, str):
            forms = [forms]

        if key not in d:
            d[key] = set()

        d[key].update(forms)

    return d

def append_unique(d, key, values):

    if key not in d:
        d[key] = set()

    if isinstance(values, str):
        d[key].add(values)
    else:
        d[key].update(values)


def singular_generator(token):
    sing_token = grammar_machine.singular_noun(token)
    if not sing_token:
        sing_token = token
    return sing_token

def plural_generator(token):
    plu_token = grammar_machine.plural_noun(token)
    if not plu_token:
        plu_token = token
    return plu_token

def possessive_generator(token):
    poss_token = grammar_machine.singular_noun(token)
    if not poss_token:
        poss_token = token
    else:
        poss_token = poss_token + "'s"
    return poss_token

def plural_possessive_generator(token):
    plu_poss_token = grammar_machine.plural_noun(token)
    if not plu_poss_token:
        plu_poss_token = token
    if plu_poss_token[-1] == 's':
        plu_poss_token = plu_poss_token + "'"
    else:
        plu_poss_token = plu_poss_token + "s'"
    return plu_poss_token

def dumb_plural_generator(token):
    #appends an s to the singular no matter what it is
    # ex. kids will say "gooses" instead of geese
    dumb_plu_token = grammar_machine.singular_noun(token)
    if not dumb_plu_token:
        dumb_plu_token = token
    dumb_plu_token = token + "s"
    return dumb_plu_token

def dumb_plural_poss_generator(token):
    #appends an s to the singular no matter what it is
    # ex. kids will say "gooses" instead of geese

    sing = grammar_machine.singular_noun(token)
    if not sing:
        sing = token

    dumb_plural = sing + "s"

    if dumb_plural.endswith("s"):
        dumb_plural_poss = dumb_plural + "'"
    else:
        dumb_plural_poss = dumb_plural + "'s"

    return dumb_plural_poss

def compound_word_finder(token):
    # word_word, word+word, word word
    cmpd_set = set()

    # for something like "french fry" to french fries
    if " " in token:
        parts = token.split()
        cmpd_set.add(" ".join(parts))
        cmpd_set.add("+".join(parts))
        cmpd_set.add("_".join(parts))

    else:
        for i in range(2, len(token)-1):  # limit split positions
            left, right = token[:i], token[i:]
            cmpd_set.update([
                f"{left} {right}",
                f"{left}+{right}",
                f"{left}_{right}"
            ])

    return cmpd_set

def grammatical_generator(alt_forms_dict, skip_list=None):

    d = alt_forms_dict.copy()

    if skip_list is None:
        skip_list = []

    # not my smartest code
    for base, alt_forms in alt_forms_dict.items():
        base_additions = set()
        for alt_form in alt_forms:
            if "plural_generator" not in skip_list:
                base_additions.add(plural_generator(alt_form))
            if "singular_generator" not in skip_list:
                base_additions.add(singular_generator(alt_form))
            if "possessive_generator" not in skip_list:
                base_additions.add(possessive_generator(alt_form))
            if "plural_possessive_generator" not in skip_list:
                base_additions.add(plural_possessive_generator(alt_form))
            if "dumb_plural_generator" not in skip_list:
                base_additions.add(dumb_plural_generator(alt_form))
            if "dumb_plural_poss_generator" not in skip_list:
                base_additions.add(dumb_plural_poss_generator(alt_form))
        alt_forms_dict[base].update(base_additions)

    if "compound_word_finder" not in skip_list:
        for base, alt_forms in alt_forms_dict.items():
            cmpd_set = set()
            for alt_form in alt_forms:
                cmpd_set |= compound_word_finder(alt_form)
            alt_forms_dict[base].update(cmpd_set)

    return alt_forms_dict


def merge_mcdi_dict_into_mcdi_df(mcdi_ibi_df, alt_form_dict, main_col='english_gloss', alt_col='alt_forms'):

    new_rows = []
    missing_keys = set()

    for _, row in mcdi_ibi_df.iterrows():
        key = row[main_col]
        alt_forms = alt_form_dict.get(key, {key})
        if isinstance(alt_forms, str):
            alt_forms = {alt_forms}
        elif not isinstance(alt_forms, (set, list)):
            alt_forms = set(alt_forms)

        for alt in alt_forms:
            new_row = row.copy()
            new_row[alt_col] = alt
            new_rows.append(new_row)

    expanded_df = pd.DataFrame(new_rows)

    expanded_df = expanded_df.drop_duplicates().reset_index(drop=True)

    expected_count = sum(len(v) if not isinstance(v, str) else 1 for v in alt_form_dict.values())
    actual_count = len(expanded_df)

    if actual_count != expected_count:
        warnings.warn(
            f"row count is mismatched. expanded_df has {actual_count} rows, "
            f"but sum of alt form lengths is {expected_count}."
        )

    if missing_keys:
        warnings.warn(f"alt_form_dict is missing these keys present in the df: {missing_keys}")

    return expanded_df