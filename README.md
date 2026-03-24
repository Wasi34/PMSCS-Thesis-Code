# Chain-of-Thought QA — Bengali SQuAD

Bengali SQuAD question-answering using **Chain-of-Thought (CoT) prompting** across 3 LLMs.

## Project Structure

```
chain_of_thought_qa/
├── gpt4omini.py          # GPT-4o mini (OpenAI API)
├── gemini25pro.py        # Gemini 2.5 Pro (Google GenAI API)
├── llama3_groq.py        # LLaMA 3.3 70B (Groq API)
├── evaluate_accuracy.py  # Accuracy evaluation (Exact Match + Partial Match)
├── .env.example          # API key template
├── requirements.txt      # Python dependencies
└── outputs/
    ├── gpt4omini/        → results.json
    ├── gemini25pro/      → results.json
    ├── llama3_groq/      → results.json
    └── evaluation_summary.json
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your keys
```

## Running the Scripts

```bash
# Test with 10 rows (default)
python gpt4omini.py
python gemini25pro.py
python llama3_groq.py

# Specify number of rows
python gpt4omini.py --rows 50
python gemini25pro.py --rows 50
python llama3_groq.py --rows 50
```

## Evaluating Accuracy

```bash
# Evaluate all models
python evaluate_accuracy.py

# Evaluate a specific model
python evaluate_accuracy.py --model gpt4omini
python evaluate_accuracy.py --model gemini25pro
python evaluate_accuracy.py --model llama3_groq
```

## Output Format (`results.json`)

Each row in the output JSON contains:

```json
{
  "id": 1,
  "title": "শেখ মুজিবুর রহমান",
  "context": "<Bengali passage>",
  "question": "<Bengali question>",
  "expected_answer": "<ground truth>",
  "cot_reasoning": "<model's step-by-step reasoning>",
  "predicted_answer": "<final answer from model>"
}
```

## Chain-of-Thought Prompt Design

All models use the same 4-step CoT system prompt:

1. **READ** — Carefully read the Bengali context  
2. **IDENTIFY** — Find the relevant part for the question  
3. **REASON** — Think step-by-step from the context  
4. **ANSWER** — Extract a concise, exact answer  

The model responds in structured JSON: `{ "reasoning": "...", "answer": "..." }`

## Dataset

- **Source**: `squad_bn - Test.csv` (Bengali SQuAD Test split, ~4300 rows)
- **Columns**: `Title`, `Context`, `Question`, `Answer_Text`
- Note: Some rows have empty `Answer_Text` — these are skipped during accuracy evaluation
