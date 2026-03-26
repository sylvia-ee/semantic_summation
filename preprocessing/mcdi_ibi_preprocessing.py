
# this .py houses all functions related to preprocessing mcdi item-by-item data
# assuming df has at minimum the columns uni_lemma and category

import pandas as pd
import re
import warnings
from collections import defaultdict
import inflect
import json

grammar_machine = inflect.engine()

# ----------------------------------------------------    
#  SETUP FUNCTIONS
# ----------------------------------------------------    

def mcdi_ibi_setup(raw_mcdi_df, 
                   orig_base="english_gloss",
                   base_col="base", 
                   cat_col="category",
                   item_id="item_id",
                   orig_sample_prod_col="24",
                   sample_prod_col="sample_prod_col"):
    
    """
    adds and renames columns to mcdi item-by-item df to denote mcdi inclusions and exclusions

    :param raw_mcdi_df: pd df of original mcdi item-by-item data, assuming at minimum cols for item id, category, production, word
    :param orig_base: str of original dataset column label for word from original df
    :param base_col: str of renamed column label for word for new df (we rename this to store the proper "cleaned" version of each word)
    :param cat_col: str of column for category from original df (we assume you won't change this)
    :param item_id: str of column for item id from original df (we assume you won't change this)
    :param orig_sample_prod_col: str of column name for sample production column from original df (i.e. average production for this word across kids) (we assume you change this)
    :param sample_prod_col: str of column for sample production column for new df 
    """

    mcdi_ibi = raw_mcdi_df.copy()

    mcdi_ibi[base_col] = mcdi_ibi[orig_base].astype(str).str.lower()
    mcdi_ibi[cat_col] = mcdi_ibi[cat_col].astype(str).str.lower()
    mcdi_ibi[item_id] = mcdi_ibi[item_id].astype(str).str.lower().astype("category")
    mcdi_ibi[sample_prod_col] = mcdi_ibi[orig_sample_prod_col].astype(float)

    mcdi_ibi = mcdi_ibi.drop_duplicates()

    mcdi_ibi['alt'] = None
    mcdi_ibi['alt_origin'] = None
    mcdi_ibi['excl_reason'] = None

    return mcdi_ibi

# ----------------------------------------------------   
#  EXCLUSION FUNCTIONS
# ----------------------------------------------------   

def exclude_cats(
    mcdi_ibi_df,
    csv_path,
    df_category_col="category",
    csv_category_col="category",
    csv_reason_col="excl_reason"
):
    """
    writes exclusion reason to exclusion column for mcdi categories we want to exclude

    :param mcdi_ibi_df: pd df of mcdi item-by-item data
    :param csv_path: path to CSV containing categories to exclude
    :param df_category_col: str for col in mcdi_ibi_df with category labels to exclude by
    :param csv_category_col: str for col in CSV containing categories to exclude
    :param csv_reason_col: str for col in CSV specifying exclusion reason 
    :return: pd df with excl_reason column filled where applicable
    """

    excl_df = pd.read_csv(csv_path)

    excl_df[csv_category_col] = excl_df[csv_category_col].astype(str).str.lower()
    mcdi_ibi_df[df_category_col] = mcdi_ibi_df[df_category_col].astype(str).str.lower()

    excl_map = dict(zip(excl_df[csv_category_col], excl_df[csv_reason_col]))

    mask = mcdi_ibi_df[df_category_col].isin(excl_map.keys())

    mcdi_ibi_df.loc[
        mask & mcdi_ibi_df["excl_reason"].isna(),
        "excl_reason"
    ] = mcdi_ibi_df.loc[
        mask & mcdi_ibi_df["excl_reason"].isna(),
        df_category_col
    ].map(excl_map)

    return mcdi_ibi_df


def exclude_proper_nouns(
    mcdi_ibi_df,
    base_col="base",
    reason_col="excl_reason"
):
    """
    writes exclusion reason "proper noun" to exclusion column to denote proper nouns we want to exclude
    primary use via call in exclude_words 

    :param mcdi_ibi_df: pd df of mcdi item-by-item data
    :param base_col: str for col in mcdi_ibi df to check for exclusions
    :param reason_col: str for col in mcdi_ibi df specifying exclusion reason
    """

    mask = mcdi_ibi_df[base_col].str.endswith(" name", na=False)

    mcdi_ibi_df.loc[
        mask & mcdi_ibi_df[reason_col].isna(),
        reason_col
    ] = "proper noun"

    return mcdi_ibi_df


def exclude_via_csv(
    mcdi_ibi_df,
    path2csv,
    df_base_col="base",
    df_alt_col="alt",
    csv_base_col="base",
    csv_alt_col="alt",
    csv_reason_col="excl_reason"
):
    """
    writes exclusions from separate CSV with cols: base, alt, excl_reason denoting exclusion to mcdi_ibi df 
    - base filled, alt blank: exclude all rows with that base
    - base blank, alt filled: exclude all rows with that alt
    - both filled: exclude exact match

    :param mcdi_ibi_df: pd df of mcdi item-by-item data
    :param path2csv: str for path to csv where exclusions are specified
    :param df_base_col: str for name of base col in mcdi_ibi_df
    :param df_alt_col: str for name of alt col in mcdi_ibi_df
    :param csv_base_col: str for name of base col in CSV
    :param csv_alt_col: str for name of alt col in CSV
    :param csv_reason_col: str for name of excl_reason col in CSV
    """

    excl_df = pd.read_csv(path2csv)

    for _, row in excl_df.iterrows():
        base = row[csv_base_col]
        alt = row[csv_alt_col]
        reason = row[csv_reason_col]

        if pd.notna(base) and pd.isna(alt):
            mask = mcdi_ibi_df[df_base_col] == base

        elif pd.notna(alt) and pd.isna(base):
            mask = mcdi_ibi_df[df_alt_col] == alt

        elif pd.notna(base) and pd.notna(alt):
            mask = (
                (mcdi_ibi_df[df_base_col] == base) &
                (mcdi_ibi_df[df_alt_col] == alt)
            )
        else:
            continue

        mcdi_ibi_df.loc[
            mask & mcdi_ibi_df["excl_reason"].isna(),
            "excl_reason"
        ] = reason

    return mcdi_ibi_df


def exclude_words(
    mcdi_ibi_df,
    exclusion_funcs=None,
    csv_paths=None,
    base_col="base",
    df_alt_col="alt",
    csv_base_col="base",
    csv_alt_col="alt",
    csv_reason_col="excl_reason"
):
    #TODO: modify this function to accept **kwargs in case functions passed want arguments 

    """
    writes exclusions from exclusion functions and CSV-based exclusions to mcdi item-by-item dataframe 
    designed this way so you only call one function to write all desired word-level exclusions for mcdi_ibi.

    :param mcdi_ibi_df: pd df of mcdi item-by-item data
    :exclusion_funcs: list of function objs for functions you want to run to perform exclusion
    :csv_paths: list of str csv paths for exclusions csvs you want to specify exclusion from (see exclude_via_csv for assumptions)
    :base_col: str for base column in mcdi item-by-item data
    :df_alt_col: str for alt column in mcdi item-by-item data
    :csv_base_col: str for base column in exclusion data csv 
    :csv_alt_col: str for alt col in exclusion data csv
    :csv_reason_col: str for excl_reason in exclusion data csv specifying exclusion reason 

    """

    if exclusion_funcs is None:
        exclusion_funcs = []

    if csv_paths is None:
        csv_paths = []

    for func in exclusion_funcs:
        mcdi_ibi_df = func(mcdi_ibi_df, base_col=base_col)

    for path in csv_paths:
        mcdi_ibi_df = exclude_via_csv(
            mcdi_ibi_df,
            path2csv=path,
            df_base_col=base_col,
            df_alt_col=df_alt_col,
            csv_base_col=csv_base_col,
            csv_alt_col=csv_alt_col,
            csv_reason_col=csv_reason_col
        )

    return mcdi_ibi_df


def strip_syntax(mcdi_ibi_df, 
                 base_col="base",
                 alt_col="alt",
                 alt_origin_col="alt_origin"):
    
    """
    strips special mcdi-word notation such as () and * at the end of words to enable downstream token-level analysis
    see additional comments inside function for specific details 

    :param mcdi_ibi_df: pd dataframe of mcdi item-by-item data
    :param base_col: str of mcdi column name containing the word to be reformatted (this should be base bc base has been lowercase-normed in setup)
    :param alt_col: str of mcdi column name containing the alt column to write the reformatted word to
    :param alt_origin_col: str of mcdi column name describing where the processed alternative form of the word "appeared" from
    """
     
    # search for cases in col uni_lemma and strip using regex.
    # word* -> word
    # word () -> word
    # word ()* -> word
    # word/word -> 2 word alt forms
    # word/word* -> 2 word alt forms
    # word/word () -> 2 word alt forms
    # word/word ()* -> 2 word alt forms

    # e.g. bottom/buttocks returns two rows, 1st row as bottom/buttocks and bottom
    #                                        2nd row as bottom/buttocks and buttocks

    stripped_dict = {}

    for base_wd in mcdi_ibi_df[base_col]:
        cleaned = re.sub(r"\*", "", base_wd).strip()
        alt_no_sense = re.sub(r"\s*\([^)]*\)\s*", "", cleaned).strip()
        alt_forms = [w.strip() for w in alt_no_sense.split('/') if w.strip()]
        stripped_dict[base_wd] = alt_forms

    new_rows = []
    for base_wd, forms in stripped_dict.items():
        for f in forms:
            new_rows.append({
                base_col: base_wd,
                alt_col: f,
                alt_origin_col: 'syntax'
            })

    alt_form_rows = pd.DataFrame(new_rows)
    mcdi_ibi_df_no_alt = mcdi_ibi_df.drop(columns=[alt_col, alt_origin_col], errors='ignore')
    syntax_cleaned_mcdi_ibi_df = mcdi_ibi_df_no_alt.merge(
        alt_form_rows, on=base_col, how='outer'
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


# ----------------------------------------------------   
#  INCLUSION FUNCTIONS
# ----------------------------------------------------  

# you may notice that the inclusion functions have a more segregated pipeline than the exclusion ones
# since inclusion tends to be for specific theoretical reasons or committments, it is kept this way on purpose
# dicts are used for efficient lookup here 

def create_alt_form_dict(mcdi_ibi_df, base_col='base', alt_col='alt'):
    """ 
    looks at the mcdi_ibi_df after preprocessing and generates a first-pass dictionary of each lemma 
    and its alternative forms to facilitate matching of single mcdi word to multiple forms.
    Stores metadata as a dict for each alt form.

    :param mcdi_ibi_df: pd df with at minimum 2 cols, one for the word, another for each alternative form that word has
    :param base_col: str of pd df column name with word to look for
    :param alt_col: str of pd df column name with alternative forms of each word

    :returns alt_map: dictionary of each word in mcdi_ibi_df as 
        {base form: {alt form: {"reason": None, "initials": None, "source": None}, ...}, ...}
    """

    # initialize dictionary
    alt_map = defaultdict(dict)

    # process original df to remove all empty rows and make a copy, then lowercase
    df = mcdi_ibi_df.dropna(subset=[alt_col]).copy()
    df[base_col] = df[base_col].str.lower()
    df[alt_col] = df[alt_col].str.lower()

    # make dictionary of each word and alt forms from df
    for _, row in df.iterrows():
        base = row[base_col]
        alt = row[alt_col]
        # initialize alt form with empty metadata
        alt_map[base][alt] = {"reason": None, "initials": None, "source": None}
    
    return dict(alt_map)  # convert to regular dict


def manual_inclusions(alt_forms_dict, csv_path, base_col="base", alt_col="alt",
                      incl_reason="incl_reason", initials="initials", source="source"):
    
    #NOTE: manual inclusions are csv only such that if LLMs were introduced to generate alts, we would be able to track back data
    #NOTE: this step MUST be performed before any downstream steps so that their inclusion is also grammatically transformed as necessary

    """
    updates existing alt_forms_dict using base/alt pairs from a CSV to include non-programmatically generated 
    alternative forms for words. Each alt form includes metadata: reason, initials, source.

    :param alt_forms_dict: existing dictionary {base: {alt: {metadata}}}
    :param csv_path: path to CSV containing manual inclusions
    :param base_col: column name in CSV for base word
    :param alt_col: column name in CSV for alternate form
    :param incl_reason: col name in CSV specifying why word was included
    :param initials: str for initials of person who approved inclusion of alt form
    :param source: str denoting where the alt form originated from (e.g. ChatGPT 4.0, initials if from a person)
    :return: updated dictionary
    """

    d = {k: dict(v) for k, v in alt_forms_dict.items()}
    base_keys = set(d.keys())

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        key = row[base_col]
        form = row[alt_col]
        reason = row.get(incl_reason, None)
        person = row.get(initials, None)
        origin = row.get(source, None)

        if key not in base_keys:
            warnings.warn(f"warning: '{key}' not an mcdi word or not the exact mcdi form. check for typos?")

        if pd.isna(key) or pd.isna(form):
            continue

        if isinstance(form, str):
            form = [form]

        if key not in d:
            d[key] = {}

        for f in form:
            if f not in d[key]:
                d[key][f] = {"reason": reason, "initials": person, "source": origin}
            else:
                # update metadata if already exists
                d[key][f].update({"reason": reason, "initials": person, "source": origin})

    return d

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

def grammatically_generated_inclusions(
    alt_forms_dict, 
    funcs_to_run=[
        (plural_generator, "plural"),
        (singular_generator, "singular"),
        (possessive_generator, "possessive"),
        (plural_possessive_generator, "plural possessive"),
        (dumb_plural_generator, "dumb plural"),
        (dumb_plural_poss_generator, "dumb plural possessive")
    ]
):

    """
    applies grammatical transforms to all alt forms in the alt forms dict (e.g. "dog" -> "dog+s" both map to "dog")

    :param alt_forms_dict: existing dictionary {base: {alt: {metadata}}}
    :param funcs_to_run: list of tuples where (function object, str with reason) for all functions to run on dict provided
    """

    d = {k: dict(v) for k, v in alt_forms_dict.items()}

    for base, alt_forms in list(d.items()):
        additions = {}

        for alt_form in alt_forms:
            for func, reason in funcs_to_run:
                generated = func(alt_form)
                if isinstance(generated, str):
                    generated = [generated]  
                elif isinstance(generated, set):
                    generated = list(generated) 
                for gen in generated:
                    if gen not in d[base] and gen not in additions:
                        additions[gen] = {"reason": reason, "initials": None, "source": "grammatical_generator"}
                    elif gen in d[base]:
                        d[base][gen].update({"reason": reason})

        d[base].update(additions)

    return d

def apply_compounding(alt_forms_dict):
    d = {k: dict(v) for k, v in alt_forms_dict.items()}

    for base, alt_forms in list(d.items()):
        additions = {}

        for alt_form in list(alt_forms.keys()):
            cmpd_set = compound_word_finder(alt_form)

            for cmpd in cmpd_set:
                if cmpd not in d[base]:
                    additions[cmpd] = {
                        "reason": "compound-childes_friendly",
                        "initials": None,
                        "source": "grammatical_generator"
                    }

        d[base].update(additions)

    return d

def merge_mcdi_incl_dict_w_mcdi_df(mcdi_ibi_df, 
                                   alt_form_dict, 
                                   base_col='base', 
                                   alt_col='alt', 
                                   reason_col='incl_reason', 
                                   initials_col='initials', 
                                   source_col='source'):
    """
    once alt forms dict is done being created, applies all inclusions to dataframe 

    :param mcdi_ibi_df: pd df of dataframe to be modified (mcdi item-by-item dataframe)
    :param alt_form_dict: dict of inclusions to be applied 
    :param base_col: str of what base col is in mcdi_ibi_df
    :param alt_col: str of what alt col is in mcdi_ibi_df 
    :param reason_col: str of what reason col is in mcdi_ibi_df 
    :param initials_col: str of what initials col is in mcdi_ibi_df 
    :param source_col: str of what source col is in mcdi_ibi_df 

    """

    new_rows = []
    missing_keys = set()

    for _, row in mcdi_ibi_df.iterrows():
        key = row[base_col]

        alt_forms_meta = alt_form_dict.get(key)

        if alt_forms_meta is None:
            missing_keys.add(key)
            alt_forms_meta = {key: {"reason": None, "initials": None, "source": None}}

        for alt, meta in alt_forms_meta.items():
            new_row = row.copy()
            new_row[alt_col] = alt
            new_row[reason_col] = meta.get("reason")
            new_row[initials_col] = meta.get("initials")
            new_row[source_col] = meta.get("source")
            new_rows.append(new_row)

    expanded_df = pd.DataFrame(new_rows)

    expanded_df = expanded_df.drop_duplicates().sort_values(by=[base_col, alt_col]).reset_index(drop=True)

    if missing_keys:
        warnings.warn(f"alt_form_dict is missing these keys present in the df: {missing_keys}")

    return expanded_df

