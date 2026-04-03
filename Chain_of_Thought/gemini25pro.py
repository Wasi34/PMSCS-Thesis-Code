"""
gemini25pro.py
--------------
Chain-of-Thought (CoT) Question Answering on Bengali SQuAD using Gemini 2.5 Pro.

Usage:
    python gemini25pro.py --rows 10

Output:
    outputs/gemini25pro/results.json
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "squad_bn - Test.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "gemini25pro")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.json")
MODEL = "gemini-2.5-pro"  # gemini-2.5-pro requires paid tier; using gemini-2.0-flash (free tier compatible)

# ── CoT System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert question-answering assistant specializing in Bengali text.

Given a Bengali passage (Context) and a Question about it, follow these steps carefully:

Step 1 - READ: Carefully read the entire context.
Step 2 - IDENTIFY: Locate the part of the context most relevant to the question.
Step 3 - REASON: Think through the answer step-by-step based only on the context.
Step 4 - ANSWER: Provide a concise, exact answer extracted directly from the context.

You MUST respond with valid JSON only, in this exact format:
{
  "reasoning": "<your detailed step-by-step reasoning here>",
  "answer": "<concise final answer extracted from context>"
}

Do not add any text outside the JSON."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Please answer the question following the chain-of-thought steps."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought QA using Gemini 2.5 Pro on Bengali SQuAD"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows to process from the test set (default: 10)",
    )
    return parser.parse_args()


def load_dataset(n_rows: int) -> pd.DataFrame:
    """Load and slice the Bengali SQuAD test dataset."""
    df = pd.read_csv(DATASET_PATH)
    df = df[["Title", "Context", "Question", "Answer_Text"]].dropna(
        subset=["Context", "Question"]
    )
    df["Answer_Text"] = df["Answer_Text"].fillna("")
    df = df.head(n_rows).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from dataset.")
    return df


def query_gemini(client: genai.Client, context: str, question: str) -> dict:
    """Send a CoT prompt to Gemini 2.5 Pro and return parsed JSON response."""
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"reasoning": raw, "answer": ""}

    return {
        "reasoning": parsed.get("reasoning", ""),
        "answer": parsed.get("answer", ""),
    }


def main():
    args = parse_args()

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load data
    df = load_dataset(args.rows)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    print(f"\nRunning CoT QA with {MODEL} on {len(df)} rows...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            response = query_gemini(
                client=client,
                context=str(row["Context"]),
                question=str(row["Question"]),
            )

            result = {
                "id": idx + 1,
                "title": str(row.get("Title", "")),
                "context": str(row["Context"]),
                "question": str(row["Question"]),
                "expected_answer": str(row["Answer_Text"]),
                "cot_reasoning": response["reasoning"],
                "predicted_answer": response["answer"],
            }

            # Small delay to respect rate limits
            time.sleep(1)

        except Exception as e:
            result = {
                "id": idx + 1,
                "title": str(row.get("Title", "")),
                "context": str(row["Context"]),
                "question": str(row["Question"]),
                "expected_answer": str(row["Answer_Text"]),
                "cot_reasoning": f"ERROR: {str(e)}",
                "predicted_answer": "",
            }
            # Back off on errors
            time.sleep(3)

        results.append(result)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to: {OUTPUT_FILE}")
    print(f"   Total rows processed: {len(results)}")

    # Quick summary
    answered = sum(1 for r in results if r["predicted_answer"].strip())
    print(f"   Rows with answers:    {answered}/{len(results)}")


if __name__ == "__main__":
    main()
