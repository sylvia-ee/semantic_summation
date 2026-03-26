from preprocessing.parse_raw_data_folder import *
from preprocessing.mcdi_ibi_preprocessing import *
from preprocessing.childes_preprocessing import *

# TODO: set up paths to be defined relative to pixi toml specs
# TODO: reformulate preprocessing to permit more flexible noun-exclusion/inclusion rather than complete eradication

# process folders and set up paths
# TODO: after finalizing pipeline, check back to see if there are organization quirks or differences that need to be implemented upstream
paths_df = process_data_folder("/Users/se/Projects/semantic_summation/data")
dfs_dict = load_sample_dfs(paths_df)

# preprocess 
for sample, associated_dfs_dict in dfs_dict.items():
    
    # preprocess mcdi_ibi 
    mcdi_ibi = associated_dfs_dict["mcdi_ibi_df"] 
    mcdi_ibi = mcdi_ibi_setup(mcdi_ibi)
    mcdi_ibi = exclude_cats(mcdi_ibi, "/Users/se/Projects/semantic_summation/data/manual_preprocessing/category-exclusions_set1.csv")
    mcdi_ibi = exclude_words(mcdi_ibi, exclusion_funcs=[exclude_proper_nouns], csv_paths=["/Users/se/Projects/semantic_summation/data/manual_preprocessing/word-exclusions_set1.csv"])
    mcdi_ibi = strip_syntax(mcdi_ibi)

    mcdi_ibi_dict = create_alt_form_dict(mcdi_ibi)
    mcdi_ibi_dict = manual_inclusions(mcdi_ibi_dict, "/Users/se/Projects/semantic_summation/data/manual_preprocessing/word-inclusions_set1.csv")
    mcdi_ibi_dict = grammatically_generated_inclusions(mcdi_ibi_dict)
    mcdi_ibi_dict = apply_compounding(mcdi_ibi_dict)
    merge_mcdi_incl_dict_w_mcdi_df(mcdi_ibi, mcdi_ibi_dict).to_csv("test2.csv")

   # mcdi_alt_form_dict = create_alt_form_dict(mcdi_ibi, main_col='english_gloss', alt_col='alt_forms')
    #mcdi_alt_form_dict = manual_inclusions(mcdi_alt_form_dict, "/Users/se/Projects/semantic_summation/data/manual_preprocessing/inclusions_set1.csv")
    #mcdi_alt_form_dict = grammatical_generator(mcdi_alt_form_dict, skip_list=None)
   # mcdi_ibi = merge_mcdi_dict_into_mcdi_df(mcdi_ibi, mcdi_alt_form_dict)

    # preprocess childes.csv
    #childes = associated_dfs_dict["childes_df"]
    #childes_tokens_dict = childes_cleaner(childes)

    


