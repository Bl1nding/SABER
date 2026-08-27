import asyncio
import json
import math
import os
import uuid
import random
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from collections import Counter

# vLLM components
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams,RequestOutputKind

from src.dataloader import load_data, build_prompt
from math_eval_tools.main import evaluate,obtain_answer,clean_answer


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class SABERBranchUQEarlyExitEngine:
    def __init__(self, engine: AsyncLLMEngine, tokenizer, args):

        self.engine = engine
        self.tokenizer = tokenizer
        self.args = args
    
        # Scoring and probing settings
        self.branch_uq_diff_threshold = args.branch_uq_diff_threshold
        self.min_step_tokens = args.min_step_tokens
        self.gamma = 3
        # Tokens that control reasoning termination
        self.think_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        
        # Neutral and adversarial probe prompts
        self.neutral_probe_ids = self.tokenizer("Wait, let me summarize. The answer is \\boxed{", add_special_tokens=False).input_ids
        self.adversarial_probe_ids = self.tokenizer("Wait, I think my previous reasoning was incorrect. After correcting it, the answer is \\boxed{", add_special_tokens=False).input_ids
        
        # Tokens that trigger a probe or end generation
        # More trigger words can be added to this list (e.g. "\n\n")
        raw_suffixes = ["Wait"]
        self.stop_suffix_ids = list(set([self.tokenizer.encode(s, add_special_tokens=False)[0] for s in raw_suffixes]))
        self.stop_suffix_ids.extend([self.think_end_token_id, self.tokenizer.eos_token_id])
    
    @staticmethod
    def compute_entropy(answers: List[str]) -> float:
        """
        Predictive Entropy over sampled answers.
        Normalize by log(n), therefore H in [0,1].
        """
        BOX = "\\boxed{"

        cleaned = [clean_answer(BOX + a + "}") for a in answers]

        counter = Counter(cleaned)

        probs = [v / len(cleaned) for v in counter.values()]

        entropy = -sum(
            p * math.log(p)
            for p in probs
            if p > 0
        )

        max_entropy = math.log(len(cleaned))

        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy
    
    
    async def _probe_branch(self, prefix_ids: List[int], suffix_ids: List[int], n: int = 4) -> List[Tuple[str, float, int]]:
        """
        Generate ``n`` samples for one probe branch.

        Each result contains the generated text, mean negative log
        probability, and generated token count.
        """
        pred_prob_stop_tokens = ['} ',' }', '}\n', '}\n\n', '}.', '}?' , '}.\n', '}\\', '}}', ')}', ')}.', ')}\n']
        
        sampling_params = SamplingParams(
            max_tokens=10,
            temperature=0.6,
            n=n,
            logprobs=2,
            output_kind=RequestOutputKind.FINAL_ONLY,
            stop=pred_prob_stop_tokens,
        )
        prompt_ids = prefix_ids[:-1] + suffix_ids
        request_id = f"probe_{uuid.uuid4()}"
        results_generator = self.engine.generate(
            prompt={"prompt_token_ids": prompt_ids},
            sampling_params=sampling_params,
            request_id=request_id
        )
        
        final_res = None
        async for res in results_generator:
            final_res = res
        
        # print(f"Raw outputs count: {len(final_res.outputs)}")
        # Collect text, uncertainty, and token count for each sample.
        results = []
        for output in final_res.outputs:
            gen_text = output.text.strip()
            actual_len = len(output.token_ids) 
            
            logprobs = output.logprobs
            if not logprobs:
                results.append((gen_text, 0.0, actual_len))
                continue
                
            sum_logprobs = 0.0
            valid_tokens = 0
            for token_id, lp_dict in zip(output.token_ids, logprobs):
                if lp_dict:
                    if token_id in lp_dict:
                        max_lp = lp_dict[token_id].logprob
                    else:
                        max_lp = max(obj.logprob for obj in lp_dict.values())

                    sum_logprobs += max_lp
                    valid_tokens += 1
                    
            mean_logprob = sum_logprobs / valid_tokens if valid_tokens > 0 else -100
            # Keep the value in log space for the branch uncertainty score.
            # ppl = math.exp(-mean_logprob)
            ppl = -mean_logprob
            results.append((gen_text, ppl, actual_len))
            
        return results

    async def generate_task(self, question: str, idx: int, sem: asyncio.Semaphore) -> Tuple[int, str, Dict]:
        async with sem:
            prompt = build_prompt(question, self.tokenizer)
            input_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
            
            max_gen_budget = self.args.max_tokens 
            generated_ids = []
            current_step_tokens_count = 0 
            total_probe_overhead = 0  # Total tokens generated by probes
            is_early_exited = False
            has_closed_think = self.args.no_probe
           
            consecutive_probe_fails = 0
        
            while len(generated_ids) < max_gen_budget:
                remaining_budget = max_gen_budget - len(generated_ids)
                if remaining_budget <= 0: break

                current_stop = [] if has_closed_think else self.stop_suffix_ids

                sampling_params = SamplingParams(
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=remaining_budget,
                    stop_token_ids=current_stop,
                )
                
                request_id = f"gen_{uuid.uuid4()}"
                this_step_ids = []

                async for request_output in self.engine.generate(
                    prompt={"prompt_token_ids": input_ids + generated_ids}, 
                    sampling_params=sampling_params,
                    request_id=request_id,
                ):
                    this_step_ids = request_output.outputs[0].token_ids
                
                generated_ids.extend(this_step_ids)
                current_step_tokens_count += len(this_step_ids)

                if len(this_step_ids) > 0 and this_step_ids[-1] == self.tokenizer.eos_token_id:
                    break

                if not has_closed_think and self.think_end_token_id in this_step_ids:
                    has_closed_think = True
                    current_step_tokens_count = 0
                    continue 

                if not has_closed_think and current_step_tokens_count >= self.min_step_tokens:
                    current_prefix = input_ids + generated_ids
                    
                    # Sample the neutral and adversarial branches concurrently.
                    n_samples = getattr(self.args, "probe_n", 4)
                    neutral_task = self._probe_branch(current_prefix, self.neutral_probe_ids, n=n_samples)
                    adversarial_task = self._probe_branch(current_prefix, self.adversarial_probe_ids, n=n_samples)
                    
                    neutral_results, adversarial_results = await asyncio.gather(
                        neutral_task, adversarial_task
                    )
                    
                    neutral_answers = [r[0] for r in neutral_results]
                    adversarial_answers = [r[0] for r in adversarial_results]
                    
                    # Track probe token overhead.
                    step_probe_len = sum(r[2] for r in neutral_results) + sum(r[2] for r in adversarial_results)
                    total_probe_overhead += step_probe_len
                    
                    # Estimate answer uncertainty with normalized entropy.
                    entropy_neutral = self.compute_entropy(neutral_answers)
                    entropy_adversarial = self.compute_entropy(adversarial_answers)
                    
                    # Estimate token uncertainty in log space.
                    log_ppl_neutral = np.mean(
                        [r[1] for r in neutral_results]
                    )

                    log_ppl_adversarial = np.mean(
                        [r[1] for r in adversarial_results]
                    )
                    
                    uq_neutral = entropy_neutral + log_ppl_neutral
                    uq_adversarial = entropy_adversarial + log_ppl_adversarial
                    
                    # Compare uncertainty across the two branches.
                    branch_uq_diff = abs(
                        uq_neutral
                        -
                        uq_adversarial
                    )

                    # tqdm.write(
                    #     f"[Probe] "
                    #     f"Idx={idx} "
                    #     f"H_neutral={entropy_neutral:.3f} "
                    #     f"H_adversarial={entropy_adversarial:.3f} "
                    #     f"logPPL_neutral={log_ppl_neutral:.3f} "
                    #     f"logPPL_adversarial={log_ppl_adversarial:.3f} "
                    #     f"UQ_neutral={uq_neutral:.3f} "
                    #     f"UQ_adversarial={uq_adversarial:.3f} "
                    #     f"BranchUQDiff={branch_uq_diff:.3f}"
                    # )
                    # Stop when the branch uncertainty gap is sufficiently small.
                    if (
                        branch_uq_diff <= self.branch_uq_diff_threshold
                    ):
                        is_early_exited = True
                        has_closed_think = True 
                        generated_ids.pop()
                        # Close the reasoning block and continue with the final answer.
                        exit_ids = self.tokenizer.encode("</think>\n\n", add_special_tokens=False)
                        generated_ids.extend(exit_ids)
                        current_step_tokens_count = 0
                        continue
                    else:
                        pass
                        
                    current_step_tokens_count = 0

            final_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return idx, final_text, {
                "is_early_exited": is_early_exited, 
                "actual_output_len": len(generated_ids),
                "total_probe_tokens": total_probe_overhead
            }


async def run_evaluation(args):
    seed = 42+args.runid
    set_seeds(42)

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(), 
        dtype="auto",
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        seed=seed
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    executor = SABERBranchUQEarlyExitEngine(engine, tokenizer, args)
    
    dataset = load_data(data_name=args.data_name, debug=args.debug)
    if args.max_example > 0: dataset = dataset[:args.max_example]
    
    total_count = len(dataset)
    sem = asyncio.Semaphore(args.concurrency)
    
    tasks = [
        asyncio.create_task(executor.generate_task(item["instruction"], i, sem)) 
        for i, item in enumerate(dataset)
    ]
    
    os.makedirs(args.save_path, exist_ok=True)
    results_map = {}

    print(f"Starting evaluation with {total_count} examples...")
    
    for task in tqdm(tasks, total=total_count, desc="Inferring"):
        idx, output_text, stats = await task
        results_map[idx] = {
            "id": idx,
            "model_output": output_text,
            "gold_answer": dataset[idx]["answer"],
            "stats": stats
        }

    print("\nInference complete. Scoring outputs...")
    sorted_results = [results_map[i] for i in range(total_count)]
    eval_is_correct, cleaned_answers, summary = evaluate(
        answers=[r["gold_answer"] for r in sorted_results],
        completions=[r["model_output"] for r in sorted_results]
    )
    
    for idx, (is_cor, ans) in enumerate(zip(eval_is_correct, cleaned_answers)):
        sorted_results[idx].update({"is_correct": is_cor, "cleaned_answer": ans})

    ee_count = sum(1 for r in sorted_results if r["stats"]["is_early_exited"])
    avg_tokens = sum(r["stats"]["actual_output_len"] for r in sorted_results) / total_count
    avg_probe = sum(r["stats"]["total_probe_tokens"] for r in sorted_results) / total_count
    summary.update({"early_exit_rate": ee_count/total_count, "avg_output_tokens": avg_tokens, "avg_probe_tokens": avg_probe})

    final_path = os.path.join(args.save_path, f"{args.model_name}_{args.data_name}_FINAL.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": summary, "results": sorted_results}, f, indent=4, ensure_ascii=False)
    
    print(f"Evaluation complete. Results saved to: {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--data_name", type=str, default="gsm8k")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--max_tokens", type=int, default=16*1024)
    parser.add_argument("--max_model_len", type=int, default=32*1024)
    parser.add_argument("--min_step_tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=64) 
    parser.add_argument("--gpu_util", type=float, default=0.9) 
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Tensor parallel size (defaults to torch.cuda.device_count())")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--branch_uq_diff_threshold", "--uq_diff_threshold",
        dest="branch_uq_diff_threshold", type=float, default=0.1,
        help="Maximum Branch-UQ Diff for the SABER scoring-function ablation",
    )
    parser.add_argument("--probe_n", type=int, default=4, help="Number of samples per probe branch")
    parser.add_argument("--max_example", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--no_probe", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--runid", type=int, default=0) 
    args = parser.parse_args()

    if args.max_model_len < args.max_tokens:
        args.max_model_len = max(args.max_tokens + 4000, 32768)

    asyncio.run(run_evaluation(args))
