import pandas as pd

# === CONFIG ===
DATA_FILE = "TrainWithDemoData.csv"
TARGET_RACE = "White"  # <-- Change this as needed
TRAIN_SIZE = 50000
TEST_SIZE = int(0.2 * TRAIN_SIZE)  # 20% of training size = 10k

# === LOAD DATA ===
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["Race"])

# === FILTER DATA ===
target_df = df[df["Race"] == TARGET_RACE]
other_df = df[df["Race"] != TARGET_RACE]

# === CHECK IF ENOUGH DATA ===
if len(target_df) < TRAIN_SIZE + TEST_SIZE:
    raise ValueError(f"Not enough '{TARGET_RACE}' samples to get {TRAIN_SIZE} train + {TEST_SIZE} test.")

if len(other_df) < TEST_SIZE:
    raise ValueError(f"Not enough 'non-{TARGET_RACE}' samples to create test_diff_race.")

# === SPLIT: SINGLE-RACE TRAIN to test same race vs mixed races ===
train_single_race_df = target_df.sample(n=TRAIN_SIZE, random_state=42)
remaining_target_df = target_df.drop(train_single_race_df.index)
test_same_race_df = remaining_target_df.sample(n=TEST_SIZE, random_state=1)

# === SPLIT: MIXED-RACE TRAIN to test mixed race vs specific race ===
remaining_df = df.drop(test_same_race_df.index)
train_mixed_df = remaining_df.sample(n=TRAIN_SIZE, random_state=42)

# === TEST SET: DIFFERENT RACE ===
available_diff_race_df = other_df.drop(train_mixed_df.index, errors='ignore')
test_diff_race_df = available_diff_race_df.sample(n=TEST_SIZE, random_state=2)

# === SAVE FILES ===
train_single_race_df.to_csv("train_single_race.csv", index=False)
test_same_race_df.to_csv("test_same_race.csv", index=False)
test_diff_race_df.to_csv("test_diff_race.csv", index=False)

train_mixed_df.to_csv("train_mixed_race.csv", index=False)
test_same_race_df.to_csv("test_target_race.csv", index=False)  # reuse
test_diff_race_df.to_csv("test_other_races.csv", index=False)  # reuse

# === SUMMARY ===
print("=== Final Sizes ===")
print(f"Train (only '{TARGET_RACE}'): {len(train_single_race_df)}")
print(f"Train (mixed races): {len(train_mixed_df)}")
print(f"Test (same race): {len(test_same_race_df)}")
print(f"Test (diff race): {len(test_diff_race_df)}")