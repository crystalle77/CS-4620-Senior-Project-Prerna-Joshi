import pandas as pd

# Load main label data
train_df = pd.read_csv("train_cheXbert.csv")

# Load demographic data
demo_df = pd.read_excel("CHEXPERT DEMO.xlsx")

# Load matching file with real paths
matching_df = pd.read_csv("matching.csv")  # Columns: csv_path, real_path

# Normalize patient IDs
train_df["PATIENT"] = train_df["Path"].apply(lambda x: x.split("/")[2].strip().lower())  # e.g., patient00001
demo_df["PATIENT"] = demo_df["PATIENT"].str.strip().str.lower()

# Merge realpath into train_df
matching_df = matching_df.rename(columns={"csv_path": "Path"})  # To match train_df
train_df = train_df.merge(matching_df, on="Path", how="left")

# Merge demographics into train_df
merged_df = train_df.merge(demo_df, on="PATIENT", how="left")

# Override 'Sex' and 'Age' with demographic data if available
merged_df["Sex"] = merged_df["GENDER"].combine_first(merged_df["Sex"])
merged_df["Age"] = merged_df["AGE_AT_CXR"].combine_first(merged_df["Age"])

# Drop raw gender/age columns
merged_df = merged_df.drop(columns=["GENDER", "AGE_AT_CXR"])

# Rename demographic columns
merged_df = merged_df.rename(columns={
    "PRIMARY_RACE": "Race",
    "ETHNICITY": "Ethnicity"
})

# Save final merged file
merged_df.to_csv("TrainWithDemoAndRealpath.csv", index=False)

# Preview
print(merged_df[["PATIENT", "real_path", "Sex", "Age", "Race", "Ethnicity"]].head())
