"""
gpt4omini_few_shot.py
---------------------
Few-Shot Question Answering on Bengali SQuAD using GPT-4o mini.

In few-shot mode the model receives 3 labelled Bengali QA examples in the
prompt, guiding it to the desired output format without explicit chain-of-thought
reasoning instructions.

Usage:
    python few_shot/gpt4omini_few_shot.py --rows 10

Output:
    outputs/gpt4omini/few_shot_results.json
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "gpt4omini")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "few_shot_results.json")
MODEL = "gpt-4o-mini"

# ── Few-Shot System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a question-answering assistant for Bengali text.

Given a Bengali passage (Context) and a Question, answer the question using
only the information in the context. Be concise and extract the answer
directly from the context.

Respond with valid JSON only, in this exact format:
{
  "answer": "<concise answer extracted from context>"
}

Do not add any text outside the JSON."""

# ── Few-Shot Examples (Bengali domain) ────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "context": (
            "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ। এর রাজধানী ঢাকা এবং এটি বিশ্বের সবচেয়ে ঘনবসতিপূর্ণ দেশগুলির মধ্যে একটি। "
            "বাংলাদেশের মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়েছিল এবং ১৬ ডিসেম্বর বিজয় দিবস হিসেবে পালিত হয়।"
        ),
        "question": "বাংলাদেশের রাজধানী কী?",
        "answer": "ঢাকা",
    },
    {
        "context": (
            "রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালের ৭ মে কলকাতায় জন্মগ্রহণ করেন। তিনি একজন বিশ্বখ্যাত কবি, সঙ্গীতজ্ঞ এবং দার্শনিক ছিলেন। "
            "১৯১৩ সালে গীতাঞ্জলি কাব্যগ্রন্থের জন্য তিনি নোবেল পুরস্কার লাভ করেন।"
        ),
        "question": "রবীন্দ্রনাথ ঠাকুর কোন বছর নোবেল পুরস্কার পান?",
        "answer": "১৯১৩",
    },
    {
        "context": (
            "পদ্মা সেতু বাংলাদেশের পদ্মা নদীর উপর নির্মিত একটি বহুমুখী সড়ক ও রেল সেতু। "
            "এটি ২০২২ সালের ২৫ জুন আনুষ্ঠানিকভাবে উদ্বোধন করা হয়। সেতুটির দৈর্ঘ্য ৬.১৫ কিলোমিটার।"
        ),
        "question": "পদ্মা সেতুর দৈর্ঘ্য কত?",
        "answer": "৬.১৫ কিলোমিটার",
    },
]


def build_messages(context: str, question: str) -> list:
    """
    Build a chat messages list with few-shot examples as alternating
    user/assistant turns followed by the live query.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        user_text = (
            f"--- Example {i} ---\n"
            f"Context:\n{ex['context']}\n\n"
            f"Question:\n{ex['question']}\n\nAnswer:"
        )
        assistant_text = json.dumps({"answer": ex["answer"]}, ensure_ascii=False)
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

    # Live query
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        }
    )
    return messages


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-Shot QA using GPT-4o mini on Bengali SQuAD"
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
    """Send a few-shot chat prompt to GPT-4o mini and return parsed JSON response."""
    messages = build_messages(context, question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
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

    print(f"\nRunning Few-Shot QA with {MODEL} on {len(df)} rows...\n")
    print(f"   Using {len(FEW_SHOT_EXAMPLES)} few-shot examples.\n")

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
                "prompting_strategy": "few_shot",
                "num_examples": len(FEW_SHOT_EXAMPLES),
            }

        except Exception as e:
            result = {
                "id": idx + 1,
                "title": str(row.get("Title", "")),
                "context": str(row["Context"]),
                "question": str(row["Question"]),
                "expected_answer": str(row["Answer_Text"]),
                "predicted_answer": "",
                "prompting_strategy": "few_shot",
                "num_examples": len(FEW_SHOT_EXAMPLES),
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
