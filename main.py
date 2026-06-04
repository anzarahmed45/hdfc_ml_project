from src.preprocessing import *

df = load_data(
    "dataset/hdfc_loan_dataset_full_enriched.csv"
)

df = handle_missing_values(df)

df = remove_duplicates(df)

df = encode_categorical_columns(df)

df = handle_outliers(df)

df.to_csv(
    "dataset/hdfc_loan_dataset_cleaned.csv",
    index=False
)

print("Cleaned dataset saved successfully")