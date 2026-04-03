"""
gpt4o_zero_shot.py
----------------------
Zero-Shot Question Answering on Bengali SQuAD using GPT-4o.
In zero-shot mode the model receives NO examples and NO explicit reasoning
instructions — it must answer purely from its pretrained knowledge and the
provided context, with only a minimal system instruction.

Usage:
    python zero_shot/gpt4o_zero_shot.py --rows 10

Output:
    outputs/gpt4o/zero_shot_results.json
"""

import argparse
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "squad_bn - Test.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "gpt4o")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zero_shot_results.json")
MODEL = "gpt-4o"

# ── Zero-Shot System Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a question-answering assistant for Bengali text.

Given a Bengali passage (Context) and a Question, answer the question using
only the information in the context. Be concise and extract the answer
directly from the context.

Respond with valid JSON only, in this exact format:
{
  "answer": "<concise answer extracted from context>"
}

Do not add any text outside the JSON."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-Shot QA using GPT-4o on Bengali SQuAD"
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


def query_gpt(client: OpenAI, context: str, question: str) -> dict:
    """Send a zero-shot prompt to GPT-4o and return parsed JSON response."""
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"answer": raw}

    return {"answer": parsed.get("answer", "")}


def main():
    args = parse_args()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load data
    df = load_dataset(args.rows)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    print(f"\nRunning Zero-Shot QA with {MODEL} on {len(df)} rows...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            response = query_gpt(
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
                "predicted_answer": response["answer"],
                "prompting_strategy": "zero_shot",
            }

        except Exception as e:
            result = {
                "id": idx + 1,
                "title": str(row.get("Title", "")),
                "context": str(row["Context"]),
                "question": str(row["Question"]),
                "expected_answer": str(row["Answer_Text"]),
                "predicted_answer": "",
                "prompting_strategy": "zero_shot",
                "error": str(e),
            }

        results.append(result)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Results saved to: {OUTPUT_FILE}")
    print(f"   Total rows processed: {len(results)}")

    answered = sum(1 for r in results if r["predicted_answer"].strip())
    print(f"   Rows with answers:    {answered}/{len(results)}")


if __name__ == "__main__":
    main()
