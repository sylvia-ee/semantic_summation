from preprocessing.parse_raw_data_folder import *
from preprocessing.mcdi_ibi_preprocessing import *
from preprocessing.childes_preprocessing import *

# TODO: set up paths to be defined relative to pixi toml specs
# TODO: reformulate preprocessing to permit more flexible noun-exclusion/inclusion rather than complete eradication

# process folders and set up paths
paths_df = process_data_folder("/Users/se/Projects/semantic_summation/data")
dfs_dict = load_sample_dfs(paths_df)

# preprocess 
for sample, associated_dfs_dict in dfs_dict.items():
    
    # preprocess mcdi_ibi (exclusions should be specified HERE as functions are called)
    ### 1. explicitly assign df to var (since I only plan on working with this df rn)
    ### 2. preprocess to clean notation/syntax, remove non-nouns, and manually include/exclude on certain criteria
    ### 3. create dictionary with each base word from mcdi and its potential alternative word forms
    ### 4. modify dictionary to include certain forms manually via .csv
    ### 5. modify dictionary to include certain forms via grammatical manipulation programmatically
    ### 6. modify mcdi_ibi using df to apply all changes to dictionary, which essentially functions as a glossary

    mcdi_ibi = associated_dfs_dict["mcdi_ibi_df"] 
    mcdi_ibi = mcdi_cleaner(mcdi_ibi) 
    mcdi_alt_form_dict = create_alt_form_dict(mcdi_ibi, main_col='english_gloss', alt_col='alt_forms')
    mcdi_alt_form_dict = manual_inclusions(mcdi_alt_form_dict, "/Users/se/Projects/semantic_summation/data/manual_preprocessing/inclusions_set1.csv")
    mcdi_alt_form_dict = grammatical_generator(mcdi_alt_form_dict, skip_list=None)
    mcdi_ibi = merge_mcdi_dict_into_mcdi_df(mcdi_ibi, mcdi_alt_form_dict)

    # preprocess childes.csv
    childes = associated_dfs_dict["childes_df"]
    childes_tokens_dict = childes_cleaner(childes)

    


