import pandas as pd

# Load leaderboard CSV
leaderboard_file = "final_leaderboard.csv"
df = pd.read_csv(leaderboard_file)

# Sort by accuracy descending
df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

# Compute ranks with ties (Kaggle-style)
ranks = []
prev_score = None
prev_rank = 0
count = 0

for score in df["Accuracy"]:
    count += 1
    if score == prev_score:
        rank = prev_rank
    else:
        rank = count
    ranks.append(rank)
    prev_score = score
    prev_rank = rank

df["Rank"] = ranks

# Map medals for ranks 1, 2, 3
def medal_by_rank(rank):
    if rank == 1:
        return "gold", "🥇"
    elif rank == 2:
        return "silver", "🥈"
    elif rank == 3:
        return "bronze", "🥉"
    else:
        return "", ""

df[["Class", "Medal"]] = df["Rank"].apply(lambda r: pd.Series(medal_by_rank(r)))

# Generate HTML
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GNN CoRA LLMs Competition Leaderboard</title>
<style>
html, body {
    height: 100%;
    margin: 0;
}
body {
    font-family: Arial, sans-serif;
    background: #f4f6f9;
    display: flex;
    flex-direction: column;
    justify-content: flex-start; /* can use center for vertical centering */
    align-items: center;         /* horizontal centering */
    padding: 40px 0;
}
h1 {
    text-align: center;
    margin-bottom: 40px;
}
table {
    border-collapse: collapse;
    width: 70%;              /* table width */
    background: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    border: 2px solid #2c3e50;
    margin: 0 auto;          /* horizontal centering */
}
th, td {
    padding: 16px;
    text-align: center;
    border: 1px solid #2c3e50;  /* cell borders */
}
th {
    background-color: #2c3e50;
    color: white;
    font-size: 16px;
}
tr:nth-child(even) { background-color: #f8f9fa; }
td:first-child { font-weight: bold; font-size: 18px; }
.gold { background-color: #FFD700; font-weight: bold; }
.silver { background-color: #C0C0C0; font-weight: bold; }
.bronze { background-color: #CD7F32; font-weight: bold; }
@media (max-width: 768px) { table { width: 90%; } }
</style>
</head>
<body>
<h1>🏆 GNN CoRA LLMs Competition Leaderboard</h1>
<table>
<thead>
<tr><th>Rank</th><th>Team</th><th>Accuracy</th></tr>
</thead>
<tbody>
"""

for _, row in df.iterrows():
    rank_display = f"{row['Medal']} {row['Rank']}" if row['Medal'] else row['Rank']
    team = row["Team"]
    acc = f"{row['Accuracy']:.2f}%"
    class_name = row["Class"]
    html_content += f'<tr class="{class_name}"><td>{rank_display}</td><td>{team}</td><td>{acc}</td></tr>\n'

html_content += """
</tbody>
</table>
</body>
</html>
"""

# Save HTML
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Leaderboard HTML updated successfully.")
