import re
import json
from collections import defaultdict
import numpy as np

from callGPT import call_gpt


def _extract_question(prompts, idx, kwargs):
    if "question" in kwargs and idx < len(kwargs["question"]):
        return kwargs["question"][idx]
    text = prompts[idx][0]["content"][-1]["text"]
    if "Please answer question: " in text:
        return text.split("Please answer question: ", 1)[-1]
    if "Question:" in text:
        return text.split("Question:", 1)[-1].strip()
    return text


def _extract_score(eval_prompt):
    attempt = 0
    while attempt < 3:
        attempt += 1
        try:
            output = re.sub(r"[^\w\s\.]", "", call_gpt(eval_prompt).strip().lower())
            output_num = output.split("score")[-1][:5]
            output = re.sub(r"[^\d.]", "", output_num)
            return float(output)
        except Exception as e:
            print("Error in gpt score parsing: ", e)
    return 0.0


def gpt_score_reward(prompts, completions, gt_answer, **kwargs):
    rewards = []
    for i in range(len(gt_answer)):
        if 'dataset' in kwargs and 'ovd' in kwargs['dataset'][i]:
            rewards.append(0.0)
            continue
        response, answer = completions[i], gt_answer[i]
        if "phase2_completion" in kwargs and i < len(kwargs["phase2_completion"]):
            response = kwargs["phase2_completion"][i]
        question = _extract_question(prompts, i, kwargs)

        reward = 0.0
        if '<answer>' in response:
            match_pattern = response.split('<answer>', 1)
            predicted_content = match_pattern[-1].split('</answer>')[0]
            prompt = f"""You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the json form "{{score: <score>}}". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
Question: {question} 
Standard answer: {answer} 
Model's answer: {predicted_content}"""

            output = _extract_score(prompt)
            reward += output
        rewards.append(reward)
    
    return rewards


def _extract_json_from_completion(completion, debug=False):
    import json
    
    errors = []
    
    if "```json" in completion:
        parts = completion.split("```json")
        for i, part in enumerate(parts[1:], 1):
            if "```" in part:
                json_str = part.split("```")[0].strip()
                try:
                    result = json.loads(json_str)
                    if debug:
                        print(f"[JSON Parse] Success from ```json block {i}")
                    return result
                except json.JSONDecodeError as e:
                    errors.append(f"```json block {i}: {e}")
    
    elif "```" in completion:
        parts = completion.split("```")
        for i in range(1, len(parts), 2):
            json_str = parts[i].strip()
            try:
                result = json.loads(json_str)
                if debug:
                    print(f"[JSON Parse] Success from generic ``` block {i}")
                return result
            except json.JSONDecodeError as e:
                errors.append(f"``` block {i}: {e}")
    
    def find_json_objects(text):
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                brace_count = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            objects.append(text[start:j+1])
                            i = j + 1
                            break
                else:
                    i += 1
            elif text[i] == '[':
                bracket_count = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '[':
                        bracket_count += 1
                    elif text[j] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            objects.append(text[start:j+1])
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1
        return objects
    
    json_objects = find_json_objects(completion)
    
    if json_objects:
        best_result = None
        for obj_str in json_objects:
            try:
                result = json.loads(obj_str)
                if isinstance(result, dict):
                    if 'triples' in result or 'graph' in result:
                        if debug:
                            print(f"[JSON Parse] Success from balanced braces (with keys)")
                        return result
                    if best_result is None:
                        best_result = result
                elif best_result is None:
                    best_result = result
            except json.JSONDecodeError as e:
                errors.append(f"Balanced braces: {e}")
        
        if best_result is not None:
            if debug:
                print(f"[JSON Parse] Success from balanced braces (fallback)")
            return best_result
    
    if debug:
        print(f"[JSON Parse] All strategies failed. Errors: {errors}")
    
    return None

def _extract_phase2_answer_text(completion: str) -> str:
    json_data = _extract_json_from_completion(completion)
    if isinstance(json_data, dict):
        ans = json_data.get("answer")
        if ans is not None:
            if isinstance(ans, str):
                return ans.strip()
            return str(ans).strip()
    return ""


def _compute_r_json(completion):
    json_data = _extract_json_from_completion(completion)
    if json_data is not None:
        return 2.0
    return 0.0


def _is_non_empty(value):
    if value is None:
        return False
    if isinstance(value, (list, dict, str)):
        return len(value) > 0
    return True


def _extract_structure_from_text(text):
    triples_pattern = r'["\']triples["\']\s*:\s*\[([^\]]*)\]'
    triples_match = re.search(triples_pattern, text)
    has_triples = triples_match is not None and len(triples_match.group(1).strip()) > 0
    
    graph_pattern = r'["\']graph["\']\s*:\s*\{([^}]*)\}'
    graph_match = re.search(graph_pattern, text)
    has_graph = graph_match is not None and len(graph_match.group(1).strip()) > 0
    
    return has_triples, has_graph


def _compute_r_structure(completion):
    json_data = _extract_json_from_completion(completion)
    
    if json_data is not None:
        has_triples = "triples" in json_data and _is_non_empty(json_data["triples"])
        has_graph = "graph" in json_data and _is_non_empty(json_data["graph"])
    else:
        has_triples, has_graph = _extract_structure_from_text(completion)
    
    if has_triples and has_graph:
        return 2.0
    elif has_triples or has_graph:
        return 0.5
    return 0.0


def _extract_nodes_from_json(json_data):
    nodes = []
    
    if json_data is None:
        return nodes
    
    if "triples" in json_data and isinstance(json_data["triples"], list):
        for triple in json_data["triples"]:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                nodes.append(str(triple[0]))
                nodes.append(str(triple[2]))
    
    if "graph" in json_data and isinstance(json_data["graph"], dict):
        for subject, edges in json_data["graph"].items():
            nodes.append(str(subject))
            if isinstance(edges, list):
                for edge in edges:
                    if isinstance(edge, dict) and "object" in edge:
                        nodes.append(str(edge["object"]))
    
    return nodes


def _is_loose_node(node_str):
    pattern = r'^\w+\[[^\]]+\]$|^\w+\([^\)]+\)$'
    return bool(re.match(pattern, str(node_str)))


def _is_strict_node(node_str):
    pattern = r'^\w+\[node\d+\]$'
    return bool(re.match(pattern, str(node_str)))


def _extract_nodes_from_text(text):
    pattern = r'\w+\[[^\]]+\]|\w+\([^\)]+\)'
    matches = re.findall(pattern, text)
    return matches


def _compute_r_node(completion):
    json_data = _extract_json_from_completion(completion)
    
    if json_data is not None:
        nodes = _extract_nodes_from_json(json_data)
    else:
        nodes = _extract_nodes_from_text(completion)
    
    if len(nodes) == 0:
        return 0.0
    
    loose_count = sum(1 for node in nodes if _is_loose_node(node))
    strict_count = sum(1 for node in nodes if _is_strict_node(node))
    
    if loose_count == 0:
        return 0.0
    
    ratio = strict_count / loose_count
    return 2.0 * min(1.0, ratio)


def _compute_r_leak_penalty(completion):
    completion_lower = completion.lower()
    
    leak_keywords = ['answer', 'final', 'result', 'conclude']
    
    for keyword in leak_keywords:
        if keyword in completion_lower:
            return 0.0
    
    return 1.0


def _compute_r_perplexity(solver_ppl, coef=0.1):
    ppl = float(solver_ppl)
    if not np.isfinite(ppl) or ppl <= 0:
        return 0.0
    return 10.0 * (1.0 / (1.0 + float(coef) * ppl))

def phase1_combined_reward(prompts, completions, gt_answer, **kwargs):
    solver_ppl = kwargs.get("solver_ppl")
    if solver_ppl is None:
        solver_ppl = [float("inf") for _ in gt_answer]

    rewards = []
    r_json_components = []
    r_structure_components = []
    r_node_components = []
    r_leak_penalty_components = []
    r_perplexity_components = []
    r_format_total_components = []
    w_ppl, w_leak, w_other = 0.1, 8.0, 3.0
    raw_other_max = 6.0
    solver_gpt_scores = kwargs.get("solver_gpt_scores")
    for i in range(len(gt_answer)):
        if "phase1_completion" in kwargs and i < len(kwargs["phase1_completion"]):
            completion = kwargs["phase1_completion"][i]
        else:
            completion = completions[i]

        r_json = _compute_r_json(completion)
        r_structure = _compute_r_structure(completion)
        r_node = _compute_r_node(completion)
        r_leak_penalty = _compute_r_leak_penalty(completion)
        ppl = float(solver_ppl[i]) if i < len(solver_ppl) else float("inf")
        r_perplexity = _compute_r_perplexity(ppl)
        r_perplexity_term = r_perplexity * (w_ppl / 10.0)
        phase2_gpt_score = 0.0
        if solver_gpt_scores is not None and i < len(solver_gpt_scores):
            phase2_gpt_score = float(solver_gpt_scores[i])
        phase2_gpt_score = max(0.0, min(1.0, phase2_gpt_score))
        leak_term = w_leak * r_leak_penalty * phase2_gpt_score
        format_other_raw = r_json + r_structure + r_node
        format_other_term = w_other * (format_other_raw / raw_other_max)
        r_json_scaled = w_other * (r_json / raw_other_max)
        r_structure_scaled = w_other * (r_structure / raw_other_max)
        r_node_scaled = w_other * (r_node / raw_other_max)
        r_leak_penalty_scaled = leak_term
        r_format_total = format_other_term + leak_term
        total_reward = r_format_total
        rewards.append(total_reward)
        r_json_components.append(r_json_scaled)
        r_structure_components.append(r_structure_scaled)
        r_node_components.append(r_node_scaled)
        r_leak_penalty_components.append(r_leak_penalty_scaled)
        r_perplexity_components.append(r_perplexity if w_ppl == 0.0 else r_perplexity_term)
        r_format_total_components.append(r_format_total)

    if "_phase1_breakdown_container" in kwargs:
        kwargs["_phase1_breakdown_container"]["r_json"] = r_json_components
        kwargs["_phase1_breakdown_container"]["r_structure"] = r_structure_components
        kwargs["_phase1_breakdown_container"]["r_node"] = r_node_components
        kwargs["_phase1_breakdown_container"]["r_leak_penalty"] = r_leak_penalty_components
        kwargs["_phase1_breakdown_container"]["r_perplexity"] = r_perplexity_components
        kwargs["_phase1_breakdown_container"]["r_format_total"] = r_format_total_components
        kwargs["_phase1_breakdown_container"]["phase1_total"] = list(rewards)
    return rewards


def _compute_phase2_r_json(completion):
    json_data = _extract_json_from_completion(completion)
    if json_data is not None:
        return 2.0
    return 0.0


def _extract_phase2_structure_from_text(text):
    reasoning_pattern = r'["\']reasoning["\']\s*:\s*\[([^\]]*)\]'
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    has_reasoning = reasoning_match is not None and len(reasoning_match.group(1).strip()) > 0
    
    answer_pattern = r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']'
    answer_match = re.search(answer_pattern, text)
    has_answer = answer_match is not None and len(answer_match.group(1).strip()) > 0
    
    return has_reasoning, has_answer


def _compute_phase2_r_structure(completion):
    json_data = _extract_json_from_completion(completion)
    
    if json_data is not None:
        has_reasoning = "reasoning" in json_data and _is_non_empty(json_data["reasoning"])
        has_answer = "answer" in json_data and _is_non_empty(json_data["answer"])
    else:
        has_reasoning, has_answer = _extract_phase2_structure_from_text(completion)
    
    if has_reasoning and has_answer:
        return 2.0
    elif has_reasoning or has_answer:
        return 0.5
    return 0.0


def _compute_phase2_r_operations(completion):
    reasoning_text = ""
    json_data = _extract_json_from_completion(completion)
    
    if json_data is not None and "reasoning" in json_data:
        if isinstance(json_data["reasoning"], list):
            reasoning_text = " ".join(str(step) for step in json_data["reasoning"])
        elif isinstance(json_data["reasoning"], str):
            reasoning_text = json_data["reasoning"]
    else:
        reasoning_pattern = r'["\']reasoning["\']\s*:\s*\[([^\]]*)\]'
        match = re.search(reasoning_pattern, completion, re.DOTALL)
        if match:
            reasoning_text = match.group(1)
    
    reasoning_upper = reasoning_text.upper()
    has_find = "[FIND]" in reasoning_upper
    has_follow = "[FOLLOW]" in reasoning_upper
    has_verify = "[VERIFY]" in reasoning_upper
    
    op_count = sum([has_find, has_follow, has_verify])
    
    if op_count == 3:
        return 2.0
    elif op_count == 2:
        return 1.0
    elif op_count == 1:
        return 0.5
    return 0.0


def _compute_phase2_r_correctness(prompts, completions, gt_answer, **kwargs):
    rewards = []
    correctness_scores = []
    
    for i in range(len(gt_answer)):
        response = completions[i]
        answer = gt_answer[i]
        
        json_data = _extract_json_from_completion(response)
        if json_data is not None and "answer" in json_data:
            predicted_answer = str(json_data["answer"]).strip().lower()
        else:
            predicted_answer = _extract_phase2_answer_text(response).lower()
        
        correct_answer = str(answer).strip().lower()
        
        if predicted_answer == correct_answer:
            rewards.append(4.0)
            correctness_scores.append(1.0)
        else:
            rewards.append(0.0)
            correctness_scores.append(0.0)
    
    return rewards, correctness_scores

def phase2_combined_reward(prompts, completions, gt_answer, **kwargs):
    w_json, w_structure, w_operations, w_correctness = 1.0, 2.0, 2.0, 4.0
    max_format_score = w_json * 2 + w_structure * 2 + w_operations * 2
    
    rewards = []
    r_json_components = []
    r_structure_components = []
    r_operations_components = []
    r_correctness_components = []
    r_format_total_components = []
    correctness_scores = []
    
    for i in range(len(gt_answer)):
        if "phase2_completion" in kwargs and i < len(kwargs["phase2_completion"]):
            completion = kwargs["phase2_completion"][i]
        else:
            completion = completions[i]
        
        r_json = _compute_phase2_r_json(completion)
        r_structure = _compute_phase2_r_structure(completion)
        r_operations = _compute_phase2_r_operations(completion)
        
        format_score = w_json * r_json + w_structure * r_structure + w_operations * r_operations
        
        json_data = _extract_json_from_completion(completion)
        if json_data is not None and "answer" in json_data:
            predicted_answer = str(json_data["answer"]).strip().lower()
        else:
            predicted_answer = _extract_phase2_answer_text(completion).lower()
        
        correct_answer = str(gt_answer[i]).strip().lower()
        
        if predicted_answer == correct_answer:
            r_correctness = w_correctness
            correctness_score = 1.0
        else:
            r_correctness = 0.0
            correctness_score = 0.0
        
        total_reward = format_score + r_correctness
        
        rewards.append(total_reward)
        r_json_components.append(r_json)
        r_structure_components.append(r_structure)
        r_operations_components.append(r_operations)
        r_correctness_components.append(r_correctness)
        r_format_total_components.append(format_score)
        correctness_scores.append(correctness_score)
    
    if "_correctness_scores_container" in kwargs:
        kwargs["_correctness_scores_container"]["scores"] = correctness_scores
    
    if "_phase2_breakdown_container" in kwargs:
        kwargs["_phase2_breakdown_container"]["r_json"] = r_json_components
        kwargs["_phase2_breakdown_container"]["r_structure"] = r_structure_components
        kwargs["_phase2_breakdown_container"]["r_operations"] = r_operations_components
        kwargs["_phase2_breakdown_container"]["r_correctness"] = r_correctness_components
        kwargs["_phase2_breakdown_container"]["r_format_total"] = r_format_total_components
        kwargs["_phase2_breakdown_container"]["phase2_total"] = list(rewards)
    
    return rewards
