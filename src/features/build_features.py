"""Build features for training and deployment."""
from __future__ import annotations

from datetime import datetime
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#create file paths
CLEAN_DATA_PATH = Path("data/processed/heart_disease_clean.csv")
FEATURE_MATRIX_PATH = Path("data/processed/heart_disease_feature_matrix.csv")
TARGET_OUTPUT_PATH = Path("data/processed/heart_disease_target.csv")
FEATURE_PREPARATION_ARTIFACT_PATH = Path("models/heart_disease_feature_preparation.pkl")
FEATURE_METADATA_PATH = Path("models/heart_disease_feature_metadata.json")

TARGET_COLUMN = "target"
DROP_COLUMNS = ["id", "num", "dataset"]
NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
BINARY_FEATURES = ['sex', 'fbs', 'exang']
ORDINAL_FEATURES = ['cp', 'restecg']
CATEGORICAL_FEATURES = BINARY_FEATURES + ORDINAL_FEATURES
REQUIRED_COLUMNS = DROP_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]

def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def validate_clean_data(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for feature preparation: {missing_columns}")
        

def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN]).copy()
    target = df[TARGET_COLUMN].copy()
    
    # print(feature_frame)
    # print(target)
    return feature_frame, target

# print(split_features_and_target(clean_data))

#Build custom mapping for ordinal features
def map_ordinal(X):
    X = pd.DataFrame(X, columns=["cp", "restecg"])

    cp_mapping = {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal": 2,
        "asymptomatic": 3,
    }

    restecg_mapping = {
        "normal": 0,
        "st-t abnormality": 1,
        "lv hypertrophy": 2,
    }

    X["cp"] = X["cp"].map(cp_mapping).fillna(-1)
    X["restecg"] = X["restecg"].map(restecg_mapping).fillna(-1)

    return X.values 


def build_feature_preparation_pipeline() -> ColumnTransformer:

    # Numeric
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    # Binary 
    binary_pipeline = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder()),
        ]
    )

    # # Ordinal Encoder
    ordinal_pipeline = Pipeline(
        steps=[
            ("mapper", FunctionTransformer(map_ordinal, feature_names_out="one-to-one")),
        ]
    )

    # Combine all
    feature_preparation = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
        ],
        remainder="drop",
    )
    
    return feature_preparation

# print(build_feature_preparation_pipeline())

def prepare_feature_matrix(feature_frame: pd.DataFrame, feature_preparation: ColumnTransformer
) -> tuple[pd.DataFrame, ColumnTransformer]:
    transformed_array = feature_preparation.fit_transform(feature_frame)
    # print(transformed_array)
    
    feature_names = feature_preparation.get_feature_names_out()
    # print(feature_names.tolist())
     
    prepared_features = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=feature_frame.index,
    )
    
    # print(prepared_features.head())
    
    return prepared_features, feature_preparation

#check the output of the feature preparation pipeline
# print(prepare_feature_matrix(clean_data, build_feature_preparation_pipeline()))

   
def save_feature_outputs(
    prepared_features: pd.DataFrame,
    target: pd.Series,
    feature_preparation: ColumnTransformer,
) -> None:

    # Validate input types
    if not isinstance(prepared_features, pd.DataFrame):
        raise TypeError("prepared_features must be a pandas DataFrame")

    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")

    if not isinstance(feature_preparation, ColumnTransformer):
        raise TypeError("feature_preparation must be a ColumnTransformer")

    # Create folders
    FEATURE_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_PREPARATION_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save Feature Matrix (X)
    prepared_features.to_csv(FEATURE_MATRIX_PATH, index=False)

    # Save Target (y)
    target_df = target.to_frame(name=TARGET_COLUMN)
    target_df.to_csv(TARGET_OUTPUT_PATH, index=False)

    # Save Pipeline (.pkl)
    with FEATURE_PREPARATION_ARTIFACT_PATH.open("wb") as file:
        pickle.dump(feature_preparation, file)

    # Create Metadata
    metadata = {
        "version": "v1.0",
        "created_at": datetime.now().isoformat(),

        "target_column": TARGET_COLUMN,
        "dropped_columns": DROP_COLUMNS,

        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,

        "rows": int(prepared_features.shape[0]),
        "column_feature_count": int(prepared_features.shape[1]),
        "prepared_feature_names": prepared_features.columns.tolist(),

        "feature_preparation_notes": [
            "Dropped id because it is an identifier, not a medical predictor.",
            "Dropped num because the binary target already captures the diagnosis outcome.",
            "Scaled numeric features to make linear models more stable and comparable.",
            "Preserved binary features as-is for interpretability.",
            "Excluded dataset/source-hospital from training to avoid location-driven predictions.",
            "Ordinal encoded ordinal clinical values (cp, restecg, slope, thal) to keep ordinal semantics.",
            "Did not create extra derived clinical features to avoid over-engineering.",
        ],
    }

    #Save Metadata (JSON)
    FEATURE_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),encoding="utf-8",)



def summarize_feature_preparation(prepared_features: pd.DataFrame, target: pd.Series) -> None:
    print("FEATURE PREPARATION SUMMARY")
    print(f"Loaded cleaned data from: {CLEAN_DATA_PATH}")
    print(f"Saved prepared feature matrix to: {FEATURE_MATRIX_PATH}")
    print(f"Saved target to: {TARGET_OUTPUT_PATH}")
    print(f"Saved feature preparation artifact to: {FEATURE_PREPARATION_ARTIFACT_PATH}")
    print(f"Saved metadata to: {FEATURE_METADATA_PATH}")
    print(f"Prepared feature matrix shape: {prepared_features.shape}")
    print("\nPrepared feature names:")
    print("\n".join(prepared_features.columns.tolist()))
    print("\nTarget distribution:")
    print(target.value_counts().sort_index().to_string())


def main() -> None:
    clean_df = load_clean_data()
    validate_clean_data(clean_df)
    feature_frame, target = split_features_and_target(clean_df)
    feature_preparation = build_feature_preparation_pipeline()
    prepared_features, feature_preparation = prepare_feature_matrix(feature_frame,feature_preparation,)
    
    ## CALL THE FUNCTIONS IN ORDER
    save_feature_outputs(prepared_features, target, feature_preparation)
    summarize_feature_preparation(prepared_features, target)


if __name__ == "__main__":
    main()



