
from pathlib import Path
import shutil
import pandas as pd

# with the data folder as an input as a path 
# assuming below dir structure and that each folder has two .csvs, 
# one of which has childes in the name for childes data, 
# one of which has mcdi in the name for mcdi data

## data
### samples
##### sample1
##### sample2

# -------------------------------------------------------------------------


def create_subfolders(sample_dir):

    # description
    ## creates "raw" and "processed" folders for each population involved

    # inputs
    ## sample_dir: a path to the specific sample's folder
    
    # outputs
    ## raw_dir: path to raw data folder
    ## processed_dir: path to processed data folder

    raw_dir = sample_dir / "raw"
    processed_dir = sample_dir / "processed"

    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    return raw_dir, processed_dir


def standardize_filename(file_path):

    # description
    ## renames a specific file to a specific naming convention based on strings in file name

    # inputs
    ## file_path: path to the specific file being renamed
    
    # outputs
    ## new_name: the new file name as a string

    name = file_path.name.lower()

    if "childes" in name:
        new_name = "childes.csv"
        return new_name
    
    elif "mcdi" in name and "ibi" in name:
        new_name = "mcdi_ibi.csv"
        return new_name
    
    elif "mcdi" in name and "cbc" in name:
        new_name = "mcdi_cbc.csv"
        return new_name
    
    else:
        return None


def move_everything_to_raw_and_rename(sample_dir):

    # description
    ## moves all non-raw and non-processed files and folders into raw folder
    ## renames raw files to be standard when applicable 

    # inputs
    ## sample_dir: a path to the specific sample's folder 

    raw_dir, processed_dir = create_subfolders(sample_dir)

    # move all to raw
    for item in sample_dir.iterdir():
        if item in (raw_dir, processed_dir):
            continue

        target = raw_dir / item.name

        if not target.exists():
            shutil.move(str(item), str(target))

    # rename files in raw
    for file in raw_dir.iterdir():

        if not file.is_file():
            continue

        new_name = standardize_filename(file)

        if new_name is None:
            continue

        target = raw_dir / new_name

        if not target.exists():
            file.rename(target)


def build_sample_paths_dict(sample_dir):

    # description
    # saves and generates dict for paths for raw data and future preprocessed data

    # inputs:
    ## sample_dir: a path to the specific sample's folder 

    raw_dir, processed_dir = create_subfolders(sample_dir)

    sample_paths_dict = {
        "sample_folder": sample_dir,
        "word2vec_sim_matrix": processed_dir / "word2vec.csv",
        "processed_childes": processed_dir / "childes_processed.csv",
        "processed_mcdi": processed_dir / "mcdi_processed.csv",
        "full_preprocessed_data": processed_dir / "all_data.csv"
    }

    for file in raw_dir.iterdir():
        if not file.is_file():
            continue

        key = file.stem 
        sample_paths_dict[key] = file

    return sample_paths_dict


def process_data_folder(data_dir):

    # inputs:
    ## data_dir: path to main data folder containing a 'samples' subfolder

    # outputs:
    ## "data_directory_structure.csv" - a three column .csv with "Sample", "File_Type", "Path" 

    data_dir = Path(data_dir)
    samples_dir = data_dir / "samples"

    rows = []

    for sample_dir in samples_dir.iterdir():

        if not sample_dir.is_dir():
            continue

        move_everything_to_raw_and_rename(sample_dir)

        sample_paths_dict = build_sample_paths_dict(sample_dir)

        sample_name = sample_dir.name

        for key, path in sample_paths_dict.items():
            rows.append({
                "Sample": sample_name,
                "File_Type": key,
                "Path": str(path)
            })

    df = pd.DataFrame(rows)

    # write to the main data directory (not inside samples/)
    output_path = data_dir / "data_directory_structure.csv"
    df.to_csv(output_path, index=False)

    return df
    return df

def load_sample_dfs(paths_df):

    # description
    ## loads raw data into distinctly named dfs per population

    # inputs:
    ## paths_df: dataframe returned by process_data_folder

    # outputs:
    ## dict: {population: {"childes_df": ..., "mcdi_ibi_df": ..., "mcdi_cbc_df": ...}}

    sample_data = {}

    for sample in paths_df["Sample"].unique():

        sample_df = paths_df[paths_df["Sample"] == sample]

        childes_df = None
        mcdi_ibi_df = None
        mcdi_cbc_df = None

        for _, row in sample_df.iterrows():
            file_type = row["File_Type"]
            path = row["Path"]

            if pd.isna(path):
                continue

            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            if file_type == "childes":
                childes_df = df
            elif file_type == "mcdi_ibi":
                mcdi_ibi_df = df
            elif file_type == "mcdi_cbc":
                mcdi_cbc_df = df

        sample_data[sample] = {
            "childes_df": childes_df,
            "mcdi_ibi_df": mcdi_ibi_df,
            "mcdi_cbc_df": mcdi_cbc_df
        }

    return sample_data
