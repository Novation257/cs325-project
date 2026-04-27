from ollama import chat
import pandas as pd
from tqdm import tqdm
import time
import os
import json
import re

# Prompt LLM and return response
def prompt(message, system_prompt, role="user", model="qwen3:4b"):
    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": role, "content": message}
        ],
    )
    return response["message"]["content"]

system_prompt = """
Analyze the following article about Nvidia and assess the article's overall sentiment and relevancy to Nvidia's performance as a company following this guide:

Sentiment:
-1.0: Extremely negative
-0.5: Moderately negative
0: Neutral sentiment
0.5: Moderately positive
1.0: Extremely positive

Relevancy:
0.0: Not relevant/impactful to Nvidia
0.5: Moderately relevant/impactful to Nvidia
1.0: Extremely relevant/impactful to Nvidia

Your assessments can be any decimal number between -1 and 1 for sentiment and between 0 and 1 for relevancy.

Finally, make a prediction of Nvidia's stock price one day and three days after the article's release based upon the article's contents and given stock data.

Your final assessment should be in JSON format using this format as an example:
{
"relevancy": 0.725,
"sentiment": -0.48,
"T+1D": 235.48,
"T+3D": 240.13
}

Don't overthink it; try to keep compute time down.

Article:
"""

# Get all txt files in the program's folder sorted alphabetically
files = sorted([f for f in os.listdir() if f.endswith(".txt")])

# Initialize progress bar, scores array, and counter
pbar = tqdm(files, desc="Processing articles", unit="file")
scores = []
i = 0

for file in pbar:
    complete = False
    num_errors = 0
    i += 1

    # Repeatedly try to complete operation
    while not complete:
        try:
            # Get file contents
            with open(file, "r", encoding="utf-8") as f:
                fc = f.read()

            # LLM interaction
            response = prompt(fc, system_prompt, model="qwen3:4b")

            # Extract JSON safely
            try:
                data = json.loads(response)
            except:
                match = re.search(r"\{.*?\}", response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group())
                    except:
                        data = {}
                else:
                    data = {}

            relevancy = data.get("relevancy", 0)
            sentiment = data.get("sentiment", 0)
            price1d = data.get("T+1D", 0)
            price3d = data.get("T+3D", 0)

            # Store results
            scores.append((relevancy, sentiment, price1d, price3d))

            # Update progress bar with latest values
            pbar.set_postfix({
                "rel": f"{relevancy:.2f}",
                "sent": f"{sentiment:.2f}",
                "T+1D": f"{price1d:.2f}",
                "T+3D": f"{price3d:.2f}",
            })

            time.sleep(0.5)  # avoid overwhelming the model
            complete = True

        # Try again if there are any errors, but print that an error occurred
        except:
            num_errors += 1
            if num_errors == 1: print("")
            print(f"An unexpected error occurred while processing article {i} ({num_errors})")
            continue

# Convert scores array to dataframe and save as csv
df = pd.DataFrame(scores, columns=["relevancy", "sentiment", "qwasr_T+1D", "qwasr_T+3D"])
df.to_csv("qwasr_scores.csv", index=False)

print("Saved to qwasr_scores.csv")