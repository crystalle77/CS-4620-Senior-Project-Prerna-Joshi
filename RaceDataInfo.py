import pandas as pd

# Load the Excel file
file_path = 'CHEXPERT DEMO.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Show all unique values under the 'primary_race' column
if 'PRIMARY_RACE' in df.columns:
    unique_races = df['PRIMARY_RACE'].dropna().unique()
    print("Unique races under 'primary_race' column:")
    for race in unique_races:
        print(f"- {race}")
else:
    print("Column 'primary_race' not found in the Excel file.")


# Show all unique values under the 'age' column
if 'AGE_AT_CXR' in df.columns:
    unique_races = df['AGE_AT_CXR'].dropna().unique()
    print("Unique ages under 'AGE_AT_CXR' column:")
    for race in unique_races:
        print(f"- {race}")
else:
    print("Column 'AGE_AT_CXR' not found in the Excel file.")



# Show all unique values under the 'gender' column
if 'GENDER' in df.columns:
    unique_races = df['GENDER'].dropna().unique()
    print("Unique ages under 'GENDER' column:")
    for race in unique_races:
        print(f"- {race}")
else:
    print("Column 'GENDER' not found in the Excel file.")