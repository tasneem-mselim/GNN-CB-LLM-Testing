import os
import time
import subprocess
from datetime import datetime
from llm_client import call_llm

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPETITION_DIR = os.path.abspath(os.path.join(BASE_DIR, "../cora-competition"))

PROMPT_FILE = os.path.join(BASE_DIR, "frozen_prompt.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "llm_run_outputs")

REPAIR_LIMIT = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD PROMPT
# =========================================================
def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

# =========================================================
# BUILD PROMPT (SAFE + TRIMMED)
# =========================================================
def build_prompt(template):

    readme_path = os.path.join(COMPETITION_DIR, "README.md")
    readme = open(readme_path, "r", encoding="utf-8").read()

    tree_raw = subprocess.getoutput(f'cd "{COMPETITION_DIR}" && dir /s /b')
    tree = "\n".join(tree_raw.split("\n")[:80])  # IMPORTANT LIMIT

    prompt = template
    prompt = prompt.replace("{{README_VERBATIM}}", readme)
    prompt = prompt.replace("{{TREE_OUTPUT}}", tree)
    prompt = prompt.replace("{{HEAD_AND_SHAPES}}", "CORA graph node classification dataset")
    prompt = prompt.replace("{{SUBMISSION_SPEC_FROM_README}}", "submission.csv: id,pred")
    prompt = prompt.replace("{{REQUIREMENTS_TXT}}", "torch, torch_geometric, numpy")

    return prompt

# =========================================================
# EXTRACT CODE
# =========================================================
# =========================================================
# EXTRACT ONLY THE <code> BLOCK
# =========================================================
def extract_code(text):

    # Case 1:
    # <code> ... </code>
    if "<code>" in text and "</code>" in text:
        code = text.split("<code>")[1].split("</code>")[0]
        return code.strip()

    # Case 2:
    # ```python ... ```
    if "```python" in text:
        code = text.split("```python")[1].split("```")[0]
        return code.strip()

    # Case 3:
    # generic triple backticks
    if "```" in text:
        code = text.split("```")[1].split("```")[0]
        return code.strip()

    # fallback
    return text.strip()
# =========================================================
# RUN CODE
# =========================================================
def run_code():
    return subprocess.run(
        ["python", "solution.py"],
        cwd=COMPETITION_DIR,
        capture_output=True,
        text=True,
        timeout=600
    )

# =========================================================
# MAIN PIPELINE (REPAIR LOOP)
# =========================================================
def main():

    start_time = time.time()

    template = load_prompt()
    base_prompt = build_prompt(template)

    messages = [{"role": "user", "content": base_prompt}]

    output_solution = os.path.join(OUTPUT_DIR, "solution.py")
    log_path = os.path.join(OUTPUT_DIR, "execution_log.txt")

    final_code = ""
    success = False

    for attempt in range(REPAIR_LIMIT):

        print(f"\n=== Attempt {attempt + 1}/{REPAIR_LIMIT} ===")

        response = call_llm(messages)

        with open(os.path.join(OUTPUT_DIR, f"attempt_{attempt}.md"), "w", encoding="utf-8") as f:
            f.write(response)

        code = extract_code(response)
        final_code = code

        # write solution into competition folder
        comp_solution = os.path.join(COMPETITION_DIR, "solution.py")
        with open(comp_solution, "w", encoding="utf-8") as f:
            f.write(code)

        # run code
        try:
            result = run_code()
        except Exception as e:
            result = None

        # log execution
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== ATTEMPT {attempt} =====\n")

            if result:
                f.write(result.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
            else:
                f.write("EXECUTION FAILED\n")

        submission_path = os.path.join(COMPETITION_DIR, "submission.csv")
        submission_exists = os.path.exists(submission_path)

        # success condition
        if result and result.returncode == 0 and submission_exists:
            print("SUCCESS ✔")
            success = True
            break

        # STRONG REPAIR FEEDBACK
        error_feedback = f"""
The previous solution FAILED.

STRICT REQUIREMENTS:
- Must use PyTorch Geometric
- Must implement GCN or GAT (NOT MLP)
- Must load CORA dataset correctly
- Must output submission.csv
- Must be CPU compatible

ERROR:
Return code: {getattr(result, 'returncode', 'TIMEOUT')}

STDERR:
{getattr(result, 'stderr', 'No stderr')}

Fix ONLY the error and preserve GNN structure.

Return:
<plan>
...
</plan>
<code>
...
</code>
"""

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": error_feedback})

        # LIMIT CONTEXT (CRITICAL)
        if len(messages) > 6:
            messages = messages[-6:]

    # save final solution
    with open(output_solution, "w", encoding="utf-8") as f:
        f.write(final_code)

    # move submission
    submission_src = os.path.join(COMPETITION_DIR, "submission.csv")
    if os.path.exists(submission_src):
        os.rename(submission_src, os.path.join(OUTPUT_DIR, "predictions.csv"))

    # results
    with open(os.path.join(OUTPUT_DIR, "results.md"), "w") as f:
        f.write(f"# Results\nSuccess: {success}\nAttempts: {attempt+1}\n")

    # summary
    with open(os.path.join(OUTPUT_DIR, "run_summary.csv"), "w") as f:
        f.write("timestamp,success,attempts,time_sec\n")
        f.write(f"{datetime.now()},{success},{attempt+1},{time.time()-start_time}\n")

    print("\n===== DONE =====")
    print("Success:", success)

# =========================================================
if __name__ == "__main__":
    main()
