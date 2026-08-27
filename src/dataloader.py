import json
from pathlib import Path


def build_prompt(input_text, tokenizer=None, base_model=False):
    if tokenizer is None or not tokenizer.chat_template or base_model:
        return "Question: " + input_text + "\nAnswer: "
    instruction_following = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [   {"role": "system", "content": instruction_following},
            {"role": "user", "content": input_text }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "evaluate_data"

DATA_PATHS = {
    "gsm8k": DATA_DIR / "gsm8k.jsonl",
    "math_500": DATA_DIR / "math500.jsonl",
    "aime_24": DATA_DIR / "aime_2024.jsonl",
    "aime_25": DATA_DIR / "aime_2025.jsonl",
    "aime2425": DATA_DIR / "aime2425.jsonl",
    "amc23": DATA_DIR / "amc23.jsonl",
    "olympiadbench": DATA_DIR / "olympiadbench.jsonl",
    "gpqa": DATA_DIR / "gpqa.jsonl",
}
def load_data(data_name="gsm8k", debug=False):

    path = DATA_PATHS.get(data_name)
    
    if path is None:
        raise ValueError(
            f"Unknown dataset: {data_name}. Available datasets: {', '.join(DATA_PATHS)}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    list_data_dict = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Limit every dataset to the same number of examples in debug mode.
            if debug and i >= 10:
                break
                
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Failed to parse line {i}: {e}")
                continue

            res = {}
            
            if data_name == "gsm8k":
                res["instruction"] = item.get("question")
                ans = str(item.get("answer", ""))
                # GSM8K places the short answer after the #### delimiter.
                res["answer"] = ans.split("####")[-1].strip() if "####" in ans else ans
                res["id"] = f"{data_name}-{i}"

            elif data_name == "math_500":
                res["instruction"] = item.get("problem")
                res["answer"] = item.get("answer")
                res["difficulty"] = item.get("level")
                res["id"] = f"{data_name}-{i}"

            elif data_name == "aime_24" or data_name == "aime2425":
                res["instruction"] = item.get("problem") 
                res["answer"] = item.get("answer")
                res["id"] = f"{data_name}-{i}"

            elif data_name == "aime_25":
                res["instruction"] = item.get("question")
                res["answer"] = item.get("answer")
                res["id"] = f"{data_name}-{i}"

            elif data_name == "deepmath":
                res["instruction"] = item.get("question")
                res["answer"] = item.get("final_answer")
                # Preserve DeepMath-specific metadata.
                res["difficulty"] = item.get("difficulty")
                res["topic"] = item.get("topic")
                res["id"] = f"{data_name}-{i}"

            elif data_name == "amc23":
                res["instruction"] = item.get("question")
                answer = item.get("answer")
                if isinstance(answer, (int, float)):
                    res["answer"] = str(answer)
                else:
                    res["answer"] = str(answer)
                res["id"] = item.get("id", f"{data_name}-{i}")

            elif "olympiadbench" in data_name:
                res["instruction"] = item.get("problem")
                res["answer"] = item.get("answer")
                res["id"] = f"{data_name}-{i}"
            
            elif data_name == "gpqa":
                res["instruction"] = item.get("problem")
                res["answer"] = item.get("answer")
                res["id"] = f"{data_name}-{i}"
            
            if res.get("instruction"):
                list_data_dict.append(res)

    print(f"Loaded dataset: {data_name}, examples: {len(list_data_dict)}")
    return list_data_dict
