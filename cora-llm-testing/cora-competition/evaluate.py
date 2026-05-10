import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import os
#from datetime import datetime

# Arguments
if len(sys.argv) < 3:
    print("Usage: python evaluate.py <submission_csv> <labels_csv>")
    sys.exit(1)

submission_csv = sys.argv[1]   # e.g., submission/team3.csv
labels_csv = sys.argv[2]       # e.g., private_data/test_labels.csv

# Extract team name from CSV filename
# team_name = os.path.splitext(os.path.basename(submission_csv))[0]
# Get GitHub username from workflow environment
github_user = os.environ.get("GITHUB_ACTOR")

if github_user is None:
    print("Error: GITHUB_ACTOR not found.")
    sys.exit(1)

team_name = github_user

# Load CSVs
submission = pd.read_csv(submission_csv)
labels = pd.read_csv(labels_csv)

# Merge and compute accuracy
merged = pd.merge(labels, submission, on="id")
acc = accuracy_score(merged["target_groundTruth"], merged["target"])
print(f"Accuracy for {team_name}: {acc*100:.2f}%")

# Update single leaderboard CSV
leaderboard_file = "final_leaderboard.csv"
#current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_entry = {
    'Team': team_name,
    'Accuracy': acc*100,
    #'Datetime': current_datetime  # NEW
}

# Read existing leaderboard or create new
# if os.path.exists(leaderboard_file):
#     leaderboard = pd.read_csv(leaderboard_file)
#     # Remove existing entry for this team
#     leaderboard = leaderboard[leaderboard['Team'] != team_name]
# else:
#     leaderboard = pd.DataFrame(columns=['Team', 'Accuracy'])
# Read existing leaderboard or create new
if os.path.exists(leaderboard_file):
    leaderboard = pd.read_csv(leaderboard_file)

    # 🔒 STRICT: Reject if user already submitted
    if team_name in leaderboard['Team'].values:
        print(f"User '{team_name}' already submitted. Only first submission allowed.")
        sys.exit(1)
else:
    leaderboard = pd.DataFrame(columns=['Team', 'Accuracy'])
    

# Add new entry
leaderboard = pd.concat([leaderboard, pd.DataFrame([new_entry])], ignore_index=True)

# Optional: sort by accuracy descending
leaderboard = leaderboard.sort_values(by='Accuracy', ascending=False)

# Save
leaderboard.to_csv(leaderboard_file, index=False)
print(f"Updated single leaderboard: {leaderboard_file}")
print("Current leaderboard:")
print(leaderboard.to_string(index=False))
