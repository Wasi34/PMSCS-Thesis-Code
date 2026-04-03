# Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your keys
```

# Running the Scripts
```bash
# Specify number of rows
python gpt4o.py --rows 50
python gemini25pro.py --rows 50
python llama3_groq.py --rows 50
```

# Evaluating Accuracy
```bash
# Evaluate all models
python evaluate_accuracy.py

# Evaluate a specific model
python evaluate_accuracy.py --model gpt4omini
python evaluate_accuracy.py --model gemini25pro
python evaluate_accuracy.py --model llama3_groq
```

# Output Format (`results.json`)

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

# Chain-of-Thought Prompt Design

All models use the same 4-step CoT system prompt:

1. **READ** — Carefully read the Bengali context  
2. **IDENTIFY** — Find the relevant part for the question  
3. **REASON** — Think step-by-step from the context  
4. **ANSWER** — Extract a concise, exact answer  

The model responds in structured JSON: `{ "reasoning": "...", "answer": "..." }`

# Dataset
Original dataset link: https://huggingface.co/datasets/csebuetnlp/squad_bn
