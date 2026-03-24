"""
llama3_groq.py
--------------
Chain-of-Thought (CoT) Question Answering on Bengali SQuAD using LLaMA 3 via Groq API.

Usage:
    python llama3_groq.py --rows 10

Output:
    outputs/llama3_groq/results.json
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "squad_bn - Test.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "llama3_groq")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.json")
MODEL = "llama-3.3-70b-versatile"  # Best available LLaMA 3 on Groq

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

Do not add any text outside the JSON. Do not wrap the JSON in markdown code blocks."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Please answer the question following the chain-of-thought steps."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought QA using LLaMA 3 (Groq) on Bengali SQuAD"
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


def query_llama(client: Groq, context: str, question: str) -> dict:
    """Send a CoT prompt to LLaMA 3 via Groq and return parsed JSON response."""
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # Deterministic for QA tasks
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)

    # Load data
    df = load_dataset(args.rows)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    print(f"\nRunning CoT QA with {MODEL} (via Groq) on {len(df)} rows...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            response = query_llama(
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

            # Small delay to respect Groq rate limits
            time.sleep(0.5)

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
            # Back off on errors (e.g., rate limit)
            time.sleep(5)

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
