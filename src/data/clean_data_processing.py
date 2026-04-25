"""Step 4: Clean the data by removing duplicates and handling missing values."""


# Use postponed evaluation of type hints (PEP 563)
# Keeps annotations as strings to avoid forward reference issues
from __future__ import annotations

from pathlib import Path

import pandas as pd

# FILE PATHS FOR INPUT AND OUTPUT
RAW_DATA_PATH = Path("data/raw/heart_disease_uci.csv")
CLEAN_DATA_PATH = Path("data/processed/heart_disease_clean.csv")

###### COLUMNS DEFINITIONS #######

# TARGET FEATURE
TARGET_SOURCE_COLUMN = "num"  
TARGET_COLUMN = "target"  
IDENTIFIER_COLUMNS = ["id"] 
EXCLUDE_COLUMNS = [TARGET_SOURCE_COLUMN, TARGET_COLUMN] + IDENTIFIER_COLUMNS

# NUMERIC FEATURES
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
INVALID_ZERO_AS_MISSING = ["trestbps", "chol", "oldpeak"] 

# CATEGORICAL FEATURES
CATEGORICAL_COLUMNS = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"] 

REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_SOURCE_COLUMN]  # All expected columns


##### DATA CLEANING FUNCTIONS #####
""" LOAD THE RAW DATA"""
def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

"""Check for required columns in the DataFrame"""
def validate_raw_data(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    else:
        print("All required columns are present in the raw data.")
      
    # print(f"\nRaw data contains {df.shape[0]} rows and {df.shape[1]} columns.")
        
        
"""Drop columns with a high percentage of missing values (e.g., >30%)."""       
def drop_highly_missing_columns(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    cleaned = df.copy()
    
    missing_fraction = cleaned.isnull().mean()  
    columns_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
    
    if columns_to_drop:
        print(f"Dropping columns with more than {threshold*100}% missing values: {columns_to_drop}")
        cleaned = cleaned.drop(columns=columns_to_drop)
    else:
        print(f"No columns have more than {threshold*100}% missing values.")
    
    return cleaned     


"""standardize text in categorical columns (strip spaces, normalize values)"""
def standardize_categorical_text(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    
    # # Filter existing categorical columns to avoid KeyError
    existing_cols = cleaned.columns.intersection(CATEGORICAL_COLUMNS)
    
    #cc
    for col in existing_cols:
        print(f"Unique values in '{col}' before cleaning: {cleaned[col].unique()}")
        
    
    # Strip whitespace and normalize case while preserving missing values.
    cleaned[existing_cols] = cleaned[existing_cols].apply(
        lambda col: col.map(lambda value: value.strip().lower() if isinstance(value, str) else value)
    )
    
    #cc
    print("--------------------------------------------------------------")
    for col in cleaned[existing_cols].columns:
        print(f"Unique values in '{col}' after cleaning: {cleaned[existing_cols][col].unique()}")
    
    # Apply normalization rules to standardize values (e.g., 'male' -> 'Male')
    normalization_rules = {
        "sex": {
            "male": "Male",
            "female": "Female",
            "Male": "Male",
            "Female": "Female",    
        },
        "dataset": {
            "cleveland": "Cleveland",
            "hungary": "Hungary",
            "switzerland": "Switzerland",
            "va long beach": "VA Long Beach",
            "cleveland": "Cleveland",
            "hungary": "Hungary",
            "switzerland": "Switzerland",
            "va long beach": "VA Long Beach",
        },
        "cp": {
            "asymptomatic": "asymptomatic",
            "atypical angina": "atypical angina",
            "non-anginal": "non-anginal",
            "typical angina": "typical angina",
        },
        "restecg": {
            "lv hypertrophy": "lv hypertrophy",
            "normal": "normal",
            "st-t abnormality": "st-t abnormality",
        }
    }

    for column, mapping in normalization_rules.items():
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(mapping).fillna(cleaned[column])
    # print(cleaned[CATEGORICAL_COLUMNS].head(5))        
    return cleaned
       

"""Create a binary target column from the original multi-class target."""
def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    # Convert 'num' > 0 to 1 (disease present), else 0
    cleaned[TARGET_COLUMN] = (cleaned[TARGET_SOURCE_COLUMN] > 0).astype(int)
    
    print(cleaned[[TARGET_SOURCE_COLUMN, TARGET_COLUMN]].head(5))
    return cleaned


"""Remove duplicate rows from the DataFrame."""
def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    
    print(f"Number of duplicate rows before removal: {df.duplicated().sum()}")
    return df.drop_duplicates().copy()


"""Fill missing values: median for numeric columns, mode for categorical."""
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    
    # #filter existing columns to avoid KeyError
    existing_numeric_cols = [col for col in NUMERIC_COLUMNS if col in cleaned.columns]
    existing_categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in cleaned.columns]
    
    print(f"Numeric columns to impute: {existing_numeric_cols}")
    print(f"Missing values before imputation:\n{cleaned[existing_numeric_cols].isnull().sum()}")
         
    #Replace invalid zeros with NA before imputation
    for column in INVALID_ZERO_AS_MISSING:
        # Set 0 values to pd.NA (missing)
        cleaned.loc[cleaned[column] <= 0, column] = pd.NA        
    
    # Impute numeric columns with median
    for column in existing_numeric_cols:
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)
        
        
        print(f"Imputed missing values in '{column}' with median: {median_value}")
        print(f"Missing values after imputation in '{column}': {cleaned[column].isnull().sum()}")
        
    
    print(f"Numeric columns to impute: {existing_categorical_cols}")
    print(f"Missing values before imputation:\n{cleaned[existing_categorical_cols].isnull().sum()}")
             
    # Impute categorical columns with mode (most frequent value)
    for column in existing_categorical_cols:
        mode_value = cleaned[column].mode(dropna=True)
        if not mode_value.empty:
            filled = cleaned[column].where(cleaned[column].notna(), mode_value.iloc[0])
            cleaned[column] = filled.infer_objects(copy=False)
            
            
            print(f"Imputed missing values in '{column}' with mode: {mode_value.iloc[0]}")
            print(f"Missing values after imputation in '{column}': {cleaned[column].isnull().sum()}")
        else:
            print(f"Warning: Could not compute mode for '{column}' (all values may be missing). No imputation performed.")

    return cleaned

"""Reorder columns to a standard order: identifiers, features, target."""
def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    exiting_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
    ordered_columns = exiting_cols + [TARGET_COLUMN]
    
    print(f"Reordered columns: {ordered_columns}")
    
    return df.loc[:, ordered_columns].copy()


"""Run the full cleaning pipeline on the DataFrame."""
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_raw_data(df)
  
    cleaned = drop_highly_missing_columns(df)
 
    cleaned = standardize_categorical_text(cleaned)
   
    cleaned = create_binary_target(cleaned)
   
    cleaned = remove_duplicate_rows(cleaned)
   
    cleaned = impute_missing_values(cleaned)
     
    cleaned = reorder_columns(cleaned)
    
    return cleaned

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)

# clean_data(load_raw_data()).head(10)

"""Save the cleaned DataFrame to CSV"""
def save_clean_data(df: pd.DataFrame, path: Path = CLEAN_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


"""Print a summary of the cleaning process."""
def summarize_cleaning(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    duplicate_count = int(raw_df.duplicated().sum())
    invalid_zero_summary = {column: int((raw_df[column] == 0).sum()) for column in INVALID_ZERO_AS_MISSING}

    print("STEP 4: DATA CLEANING SUMMARY")
    print(f"Raw shape: {raw_df.shape}")
    print(f"Loaded raw data from: {RAW_DATA_PATH}")
    print(f"Cleaned shape: {clean_df.shape}")
    print(f"Saved cleaned data to: {CLEAN_DATA_PATH}")
    print("--------------------------------------------------------------")
    print(f"Missing values in raw data > 30% threshold: {raw_df.isnull().mean()[raw_df.isnull().mean() > 0.3].to_dict()}")
    print(f"Dropped columns with >30% missing values: {set(raw_df.columns) - set(clean_df.columns)}")
    print(f"Missing values in cleaned data: {clean_df.isnull().sum().to_dict()}")
    print("--------------------------------------------------------------")
    print(f"Removed duplicates: {duplicate_count}")
    print("--------------------------------------------------------------")
    print("Invalid values converted to missing before imputation:")
    for column, count in invalid_zero_summary.items():
        print(f"- {column}: {count}")
    print(f"\nMissing values after cleaning: {clean_df.isna().sum().to_string()}")
    print("--------------------------------------------------------------")
    print("\nTarget distribution:")
    print(clean_df[TARGET_COLUMN].value_counts().sort_index().to_string())
    
    
"""Main function: Load, clean, save, and summarize the data."""
def main() -> None:
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_clean_data(clean_df)
    summarize_cleaning(raw_df, clean_df)


if __name__ == "__main__":
    main()
