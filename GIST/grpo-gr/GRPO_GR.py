import os, sys
if 'WANDB_PROJECT' not in os.environ:
    os.environ["WANDB_PROJECT"] = "grpo_tool_undefined"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trl'))

from rewards import (
    phase1_combined_reward,
    phase2_combined_reward,
)
from GRPO_GRTrainer import GRPOGRTrainer

from accelerate import Accelerator
from relationReasoningDataset import RelationReasoningDataset
accelerator = Accelerator()
import shutil
import numpy as np

from transformers import (
    HfArgumentParser
)
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    GRPOConfig, 
    ScriptArguments,
    get_peft_config,
)

@dataclass
class VLToolGRPOConfig(GRPOConfig):
    eval_only: bool = field(default=False, metadata={"help": "Whether to only run evaluation."})
    setting: str = field(default="rr_use_external_grounding_tool", metadata={"help": "Experiment setting."})
    project_root_path: str = field(default="", metadata={"help": "The root path of the project."})
    python_path_for_dino: str = field(default="/usr/local/anaconda3/envs/grpo_env/bin/python", metadata={"help": "The conda environment path."})
    train_data_path: str = field(default="", metadata={"help": "Path to training data."})
    train_image_folder_path: str = field(default="", metadata={"help": "Path to training images."})
    eval_data_path: str = field(default="", metadata={"help": "Path to evaluation data."})
    eval_image_folder_path: str = field(default="", metadata={"help": "Path to evaluation images."})
    max_turns: int = field(default=2, metadata={"help": "Max conversational turns."})
    tool_port_starting_num: int = field(default=8020, metadata={"help": "Starting port number for tools."})
    phase2_no_image_steps: int = field(
        default=0,
        metadata={"help": "For first N training steps, phase2 receives no image (text + question + graph only)."},
    )
    phase2_num_generations: int = field(
        default=0,
        metadata={"help": "Number of phase2 rollouts per phase1 graph. <=0 means fallback to num_generations."},
    )
    eval_num_generations: int = field(
        default=0,
        metadata={"help": "Number of eval rollouts per prompt. <=0 means fallback to num_generations."},
    )
    force_image_size: int = field(default=448, metadata={"help": "Forced resize image dimension."})
    use_backbone_lora: int = field(default=0, metadata={"help": "LoRA rank for vision backbone."})
    use_llm_lora: int = field(default=0, metadata={"help": "LoRA rank for LLM."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Freeze vision backbone parameters."})
    freeze_llm: bool = field(default=False, metadata={"help": "Freeze LLM parameters."})
    unfreeze_vit_layers: int = field(default=0, metadata={"help": "Number of ViT layers to unfreeze from the end."})
    freeze_mlp: bool = field(default=False, metadata={"help": "Freeze MLP layers."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "Unfreeze the language model head."})
    conv_style: str = field(default="internvl2_5", metadata={"help": "Convolution style for the model."})
    down_sample_ratio: float = field(default=0.5, metadata={"help": "Downsample ratio for the model."})

PHASE1_PROMPT = """You are a structured information extractor for Visual Question Answering (VQA). Given an image and a question, your task is to extract information relevant to the question and represent it as "triples + a graph", so that a downstream model can answer the question using only the graph and the question (as if it had seen the image).

--------------------------------
[General Requirements]
--------------------------------
- Extract only information that is directly relevant to the question; do not include irrelevant content
- Do not hallucinate objects or relations that are not present in the image
- Do NOT provide the final answer
- The output must be strictly valid JSON (no extra text or explanations)

--------------------------------
[Node Requirements]
--------------------------------
Nodes must belong to the following categories (examples are illustrative, not restrictive):

[Entity] real-world objects or people, e.g., man, woman, dog, car, bicycle, tree  
[Geometric] geometric elements, e.g., point_A, line_1, angle_ABC, triangle_1  
[Chart] chart elements, e.g., bar_2019, line_sales, point_Q1, category_A  
[Numeric] numerical values, e.g., value_10, value_50%  
[Attribute] descriptive properties, e.g., red, big, small  

All nodes must have unique IDs in the format:
type[nodeX], e.g.,
man[node1], bicycle[node2], red[node3]

--------------------------------
[Relation (Edge / Predicate) Requirements]
--------------------------------
Prefer using the following standard predicate set:

[Attribute]: is_a, has_color, has_shape, has_value, has_label  
[Spatial]: left_of, right_of, above, below, inside, on, under, near  
[Interaction]: holding, riding, wearing, using, connected_to  
[Quantitative]: greater_than, less_than, equal_to, approximately, rank_higher  
[Geometric/Structural]: parallel_to, perpendicular_to, intersects, forms_angle_with, has_length, has_angle, part_of, increasing, decreasing  

You may use semantically equivalent natural expressions (e.g., next_to, beside, sitting_on), but they must:
- Be semantically mappable to the standard predicates
- Not be vague or meaningless
- Prefer standard predicates when possible

--------------------------------
[Step 1: Generate Triples]
--------------------------------
Extract a set of triples (subject, predicate, object)

Requirements:
- Include only information necessary to answer the question
- Triples must support a reasoning path from the question entities to potential answer entities
- All important attributes (e.g., color, size, numerical value) must be explicitly represented
- Do not include duplicate or conflicting triples
- If a relation is uncertain from the image, do not include it

--------------------------------
[Step 2: Build Graph]
--------------------------------
Construct an adjacency list graph from the triples

Strict constraints:
- The graph must be fully derived from the triples
- Do not add relations not present in the triples
- Do not omit any relations from the triples
- Node names and IDs must exactly match those in the triples
- The graph must contain a complete reasoning path (no missing key links)
- Do not leave important nodes disconnected

--------------------------------
[Forbidden Behaviors]
--------------------------------
- Do NOT directly output the answer node (e.g., only "dog[nodeX]")
- Do NOT create shortcut structures that bypass reasoning
- Do NOT introduce objects or relations not grounded in the image

--------------------------------
[Output Format (Strict JSON)]
--------------------------------
{
  "triples": [
    ["subject[nodeX]", "predicate", "object[nodeY]"],
    ...
  ],
  "graph": {
    "subject[nodeX]": [
      {"predicate": "predicate", "object": "object[nodeY]"},
      ...
    ],
    ...
  }
}
"""

PHASE2_PROMPT = """You are a graph-based reasoning assistant. Given a Graph and a Question, you must reason step by step and produce the answer.

Use the graph as your primary source of information. Only consult the image if the graph is insufficient, but reasoning should rely on nodes and relations in the graph as much as possible.

--------------------------------
[Allowed Operations]
--------------------------------

You can ONLY use these three operations:

1. [Find]
Find nodes in the graph that satisfy certain conditions.

- Use graph nodes or predicates as conditions
- Must explicitly mention node IDs or predicates

2. [Follow]
From a given node, follow a predicate to reach another node.

- Must specify the starting node (node ID)
- Must specify the predicate
- The target node must exist in the graph

3. [Verify]
Process or confirm results, such as:
- Determine the type of a node
- Check whether a condition is satisfied
- Produce the final answer

--------------------------------
[Reasoning Rules]
--------------------------------

- Reason step by step
- Use as many steps as needed to complete the reasoning. Do not stop early if the reasoning is incomplete
- Each step should make progress toward answering the question
- Each step must use exactly ONE operation: [Find], [Follow], or [Verify]
- Each step must explicitly include node IDs (e.g., woman[node1])
- Do NOT invent nodes or relations not in the graph
- Multi-hop [Follow] is allowed
- All three operations must appear at least once

--------------------------------
[Output Format: JSON only]
--------------------------------

{
  "reasoning": [
    "Step 1: [Find] ... (must include node IDs)",
    "Step 2: [Follow] ... (must include node IDs and predicate)",
    "Step 3: [Verify] ...",
    ...
  ],
  "answer": "final answer (concise)"
}
"""

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, VLToolGRPOConfig, ModelConfig)) 
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    if accelerator.is_main_process:
        
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
        
    peft_config = get_peft_config(model_args)
    prompt_suffix = ''
    prompt = ''
    if script_args.dataset_name == 'rr':
        assert training_args.max_turns == 2, "Two-phase training requires max_turns=2."
        
        prompt = PHASE1_PROMPT
        prompt_suffix = ""
        training_args.stop_strings = ["</graph>", "</answer>"] if hasattr(training_args, "stop_strings") else None

        tools = []
        
        
        
        if ',' in training_args.train_data_path:
            assert ',' in training_args.train_image_folder_path, "Please provide the same number of image folders as the number of train data paths."
            train_dataset = {}
            train_data_paths = training_args.train_data_path.split(',')
            train_image_folder_paths = training_args.train_image_folder_path.split(',')

            train_dataset = RelationReasoningDataset(train_data_paths, train_image_folder_paths, prompt = prompt, prompt_suffix = prompt_suffix)
        else:
            train_dataset = RelationReasoningDataset(training_args.train_data_path, training_args.train_image_folder_path, prompt = prompt, prompt_suffix = prompt_suffix)
        
        if ',' in training_args.eval_data_path:
            assert ',' in training_args.eval_image_folder_path, "Please provide the same number of image folders as the number of eval data paths."
            eval_dataset = {}
            eval_data_paths = training_args.eval_data_path.split(',')
            eval_image_folder_paths = training_args.eval_image_folder_path.split(',')
            for i, eval_data_path in enumerate(eval_data_paths):
                eval_dataset[eval_data_path] = RelationReasoningDataset(eval_data_path, eval_image_folder_paths[i], prompt = prompt, prompt_suffix = prompt_suffix)
        else:
            eval_dataset = RelationReasoningDataset(training_args.eval_data_path, training_args.eval_image_folder_path, prompt = prompt, prompt_suffix = prompt_suffix, limits=None)
        
        reward_funcs = [phase2_combined_reward, phase1_combined_reward]



    else:
        raise ValueError(f"Dataset name in {training_args.setting} is not supported.")    
    
    end_flag = False
    if os.path.exists(training_args.output_dir):
        checkpoint_list = [d for d in os.listdir(training_args.output_dir) if d.endswith('end_of_training.txt')]
        if len(checkpoint_list) > 0:
            print(f"Training has been finished. Please remove {training_args.output_dir} to continue training.")
            end_flag = True
        

    if not end_flag:
        trainer = accelerator.prepare(GRPOGRTrainer( 
            args=training_args,
            model=  model_args.model_name_or_path, 
            tool_names = tools,
            reward_funcs= reward_funcs, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        ))

        if training_args.eval_only:
            trainer.evaluate()
        else:
            trainer.train()

        with open(os.path.join(training_args.output_dir, "end_of_training.txt"), "w") as f:
            f.write("Training finished.\n")
            
        
