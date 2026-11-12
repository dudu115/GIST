
import json
import importlib
import os
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch
import numpy as np
import re
import base64
from PIL import Image
import io
import time
import datetime
import random
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,

)
from vision_process import process_vision_info
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import types

if is_vllm_available():
    from vllm import LLM, SamplingParams

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from internVL_src import *


def load_tool(tool_path: str, **_kwargs):
    tool_config_path = os.path.join(tool_path, "tool_config.json")
    with open(tool_config_path, "r") as f:
        tool_cfg = json.load(f)
    tool_class = tool_cfg["tool_class"]
    module_name, class_name = tool_class.rsplit(".", 1)

    if tool_path not in sys.path:
        sys.path.insert(0, tool_path)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


class RepeatRandomSampler(Sampler):
    def __init__(self, data_source: Sized, repeat_count: int, shuffle: bool = True):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            base_indexes = torch.randperm(self.num_samples).tolist()
        else:
            base_indexes = list(range(self.num_samples))
        indexes = [idx for idx in base_indexes for _ in range(self.repeat_count)]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

class GRPOGRTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        tool_names: list[str],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        max_pixels: Optional[int] = 256*28*28,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        tool_model_id: Optional[str] = 'IDEA-Research/grounding-dino-base',
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if attn_implementation == "flash_attention_2" and "torch_dtype" not in model_init_kwargs:
            model_init_kwargs["torch_dtype"] = torch.bfloat16
        def _load_optimizer_and_scheduler(self, checkpoint_path):
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.accelerator.device))
                print("Optimizer state loaded from checkpoint.")

            if os.path.exists(scheduler_path):
                self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.accelerator.device))
                print("Scheduler state loaded from checkpoint.")

        assert isinstance(model, str), f"model must be a string, but got {type(model)}"
        model_id = model
        self.model_id = model_id
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        if isinstance(model, str) and "qwen3" in model.lower():
            model_init_kwargs.pop("use_cache", None)
        
        if "qwen" in model_id.lower():
            model = AutoModelForVision2Seq.from_pretrained(model, **model_init_kwargs)
        elif "internvl" in model_id.lower():
            config = InternVLChatConfig.from_pretrained(model_id)
            if config.llm_config.model_type == 'internlm2':
                config.llm_config.attn_implementation = 'flash_attention_2'
                print('Using flash_attention_2 for InternLM')
            else:
                config.llm_config._attn_implementation = 'flash_attention_2'
                print('Using flash_attention_2 for LLaMA')
            assert config.template == args.conv_style
            model = InternVLChatModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, config=config)
            patch_size = model.config.vision_config.patch_size
            if model.config.vision_config.image_size != args.force_image_size:
                print(f'Resizing position embedding from '
                            f'{model.config.vision_config.image_size} '
                            f'to {args.force_image_size}...')
                model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                        new_size=args.force_image_size,
                                                        patch_size=patch_size)
                model.config.vision_config.image_size = args.force_image_size
            model.config.force_image_size = args.force_image_size
            model.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))
            
        if is_deepspeed_zero3_enabled():
            if "qwen" in model_id.lower():
                self.ref_model = AutoModelForVision2Seq.from_pretrained(model_id, **model_init_kwargs)
            elif "internvl" in model_id.lower():
                self.ref_model = InternVLChatModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, config=config)
                if self.ref_model.config.vision_config.image_size != args.force_image_size:
                    print(f'Resizing position embedding from '
                                f'{self.ref_model.config.vision_config.image_size} '
                                f'to {args.force_image_size}...')
                    self.ref_model.vision_model.resize_pos_embeddings(old_size=self.ref_model.config.vision_config.image_size,
                                                            new_size=args.force_image_size,
                                                            patch_size=patch_size)
                    self.ref_model.config.vision_config.image_size = args.force_image_size
                self.ref_model.config.force_image_size = args.force_image_size
                self.ref_model.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))        
        else:
            self.ref_model = create_reference_model(model)

        if processing_class is None:
            if "qwen" in model_id.lower():
                try:
                    processing_class = AutoProcessor.from_pretrained(model_id)
                except Exception:
                    try:
                        processing_class = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
                    except Exception:
                        processing_class = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
                
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
                processing_class.tokenizer.padding_side = "left"
            elif "internvl" in model_id.lower():
                tokenizer_path = model_id
                print(f'Loading Tokenizer: {tokenizer_path}')
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
                tokenizer.tokenizer_path = tokenizer_path
                tokenizer.model_max_length = args.max_prompt_length
                token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                            QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                            REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
                num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
                img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                tcs_loader = None
                processing_class = tokenizer
                if num_new_tokens > 0:
                    model.language_model.resize_token_embeddings(len(tokenizer))
                    self.ref_model.language_model.resize_token_embeddings(len(tokenizer))
                    output_embeddings = model.language_model.get_output_embeddings().weight.data
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg
                    model.config.llm_config.vocab_size = len(tokenizer)
                    model.language_model.config.vocab_size = len(tokenizer)      
                    self.ref_model.config.llm_config.vocab_size = len(tokenizer)
                    self.ref_model.language_model.config.vocab_size = len(tokenizer)      
                model.img_context_token_id = img_context_token_id
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                self.ref_model.img_context_token_id = img_context_token_id
                self.ref_model.language_model.config.use_cache = False
                self.ref_model.vision_model.gradient_checkpointing = True
                self.ref_model.vision_model.encoder.gradient_checkpointing = True
                    
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.phase2_num_generations = int(getattr(args, "phase2_num_generations", 0))
        if self.phase2_num_generations <= 0:
            self.phase2_num_generations = self.num_generations
        self.eval_num_generations = int(getattr(args, "eval_num_generations", 0))
        if self.eval_num_generations <= 0:
            self.eval_num_generations = self.num_generations
        self.use_vllm = args.use_vllm
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )
        if self.phase2_num_generations < 1:
            raise ValueError("phase2_num_generations must be >= 1.")

        assert self.use_vllm == False
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=processing_class.pad_token_id,
        )
        self.model_accepts_loss_kwargs = False

        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.max_turns = self.args.max_turns
        self.phase2_no_image_steps = int(getattr(self.args, "phase2_no_image_steps", 0))
        self.max_tool_response = 30
        self.tool_model_id = tool_model_id 
        self.graph_two_phase = "graph" in self.args.setting or "two_phase" in self.args.setting

        self.process_uinque_id = int(self.accelerator.process_index) + self.args.tool_port_starting_num
        self.tools = {}
        self.tool_func = None
        if tool_names:
            assert len(tool_names) == 1, "Only one tool is supported for now"
            self.tool_func = tool_names[0].split(":")[0]
            for tool_name in tool_names:
                assert len(tool_name.split(":")) == 2
                tool_func = tool_name.split(":")[0]
                tool_path = tool_name.split(":")[1]
                self.tools[tool_func] = load_tool(
                    f"./custom_tools/{tool_path}",
                    port=self.process_uinque_id,
                    project_root_path=self.args.project_root_path,
                    python_path_for_dino=self.args.python_path_for_dino,
                )

        
        to_phrase = "Wait, I need to think again. "
        from_phrase = "<rethink>\n"
        if "qwen" in model_id.lower():
            self.eos_token_id = processing_class.tokenizer.eos_token_id
            self.pad_token_id = processing_class.tokenizer.pad_token_id
        elif "internvl" in model_id.lower():
            self.eos_token_id = processing_class.convert_tokens_to_ids(EOS_TOKEN)
            self.pad_token_id = processing_class.convert_tokens_to_ids(END_OF_TEXT_TOKEN)

        self.is_train = True

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["message"]

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return RepeatRandomSampler(dataset, self.num_generations, shuffle=True)

    def _get_eval_sampler(self, eval_dataset=None) -> Sampler:
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        return RepeatRandomSampler(dataset, self.eval_num_generations, shuffle=False)

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep):
        if "qwen" in self.model_id.lower():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        elif "internvl" in self.model_id.lower():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_flags=image_grid_thw)
        logits = outputs.logits
        logits = logits[:, -logits_to_keep-1:-1, :]

        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)

    def _align_qwen_vl_input_ids_attention_mask(self, prompt_inputs: dict) -> None:
        ids = prompt_inputs.get("input_ids")
        am = prompt_inputs.get("attention_mask")
        if ids is None or am is None or ids.shape[-1] == am.shape[-1]:
            return
        L = min(ids.shape[-1], am.shape[-1])
        pad_side = getattr(self.processing_class.tokenizer, "padding_side", "right")
        if pad_side == "left":
            prompt_inputs["input_ids"] = ids[:, -L:]
            prompt_inputs["attention_mask"] = am[:, -L:]
        else:
            prompt_inputs["input_ids"] = ids[:, :L]
            prompt_inputs["attention_mask"] = am[:, :L]

    def _qwen_vl_generate_inputs(self, prompt_inputs) -> dict:
        d = {k: v for k, v in prompt_inputs.items() if not str(k).startswith("_")}
        self._align_qwen_vl_input_ids_attention_mask(d)
        ids, am = d.get("input_ids"), d.get("attention_mask")
        if ids is not None and am is not None and ids.shape[-1] != am.shape[-1]:
            L = min(ids.shape[-1], am.shape[-1])
            pad_side = getattr(self.processing_class.tokenizer, "padding_side", "left")
            if pad_side == "left":
                d["input_ids"] = ids[:, -L:].contiguous()
                d["attention_mask"] = am[:, -L:].contiguous()
            else:
                d["input_ids"] = ids[:, :L].contiguous()
                d["attention_mask"] = am[:, :L].contiguous()
        allowed = (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        )
        out = {k: d[k] for k in allowed if k in d and d[k] is not None}
        if "qwen3" in self.model_id.lower() and "input_ids" in out:
            pad_id = getattr(self.processing_class.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                out["attention_mask"] = (out["input_ids"] != pad_id).long()
            else:
                out["attention_mask"] = torch.ones_like(out["input_ids"], dtype=torch.long, device=out["input_ids"].device)
        return out
   
    def parse_grounding(self, text):
        pattern = r'\b\d+,\s*\d+,\s*\d+,\s*\d+\b'
        matches = re.findall(pattern, text)

        if len(matches)>0:
            return matches
        else:
            return None

    def _extract_graph_json(self, text: str) -> str:
        try:
            body = text.split("</graph>")[0].strip()
            first = body.find("{")
            last = body.rfind("}")
            if first == -1 or last == -1 or first >= last:
                return "{}"
            parsed = json.loads(body[first:last + 1])
            sanitized = {}
            if isinstance(parsed, dict):
                if "triplets" in parsed:
                    sanitized["triplets"] = parsed["triplets"]
                if "graph" in parsed:
                    sanitized["graph"] = parsed["graph"]
            if not sanitized:
                sanitized = {"graph": parsed.get("graph", {})} if isinstance(parsed, dict) else {"graph": {}}
            return json.dumps(sanitized, ensure_ascii=True)
        except Exception:
            return "{}"

    def _build_turn2_user_message(self, question_text: str, graph_json: str, image_path: Optional[str] = None):
        from GRPO_GR import PHASE2_PROMPT as PHASE2_SOLVER_PROMPT
        text = (
            f"{PHASE2_SOLVER_PROMPT}\n\nQuestion: {question_text}\nGraph_JSON: {graph_json}\n"
            "Do not rely on any hidden previous assistant text. Use only Graph_JSON and optional image."
        )
        content = [{"type": "text", "text": text}]
        if image_path is not None:
            content.insert(0, {"type": "image", "image": image_path})
        return {"role": "user", "content": content}
        
    def multi_modal_get_item(self, data_items):
        pixel_values = []
        processed_conversations = []
        num_patches = []
        images = []
        for data_item in data_items:
            image_path = [data_item['message'][i]['content'][k]['image'] for i in range(len(data_item['message'])) for k in range(len(data_item['message'][i]['content'])) if 'image' in data_item['message'][i]['content'][k]]
            num_image = len(image_path)
            assert len(image_path) == 1, f'The number of image input should be 1, but got {len(image_path)}.'
            
            pixel_value, image = load_image(image_path[0], self.is_train)
            images.append(image)
            pixel_values.append(pixel_value)

            num_patches.append(pixel_value.size(0))


            conversations = [
                {'from': 'human', 'value': '<image>\n' + data_item['message'][0]['content'][-1]['text']},
            ]
            
            
            template_name = self.args.conv_style
            num_image_token_list = [self.model.num_image_token * num_patches[-1]]
            
            if conversations[0]['from'] == 'system':
                system_prompt = conversations[0]['value']
                conversations = conversations[1:]
            else:
                conv = get_conv_template(template_name)
                system_prompt = conv.system_message
            new_conversations = []
            for conversation in conversations:
                if conversation['from'] == 'human':
                    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[0]}{IMG_END_TOKEN}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                new_conversations.append(conversation)
            conversations = new_conversations

            batches, roles = [], []
            if system_prompt is not None:
                batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
                roles.append('system')
            for conversation in conversations:
                if conversation['from'] == 'human':
                    batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
                    roles.append('human')
                elif conversation['from'] == 'gpt':
                    batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
                    roles.append('gpt')
                else:
                    raise NotImplementedError
            processed_conversations.append(''.join(batches))
        
        chosen_ret = self.processing_class(
                    processed_conversations,
                    padding=True,
                    padding_side='left',
                    return_tensors="pt",
                    max_length=self.processing_class.model_max_length,
                    truncation=True,
                )
        
        ret = dict(
            input_ids=chosen_ret['input_ids'],
            attention_mask=chosen_ret['attention_mask'],
            pixel_values=torch.stack([sect for sects in pixel_values for sect in sects],dim=0),
            image_grid_thw=torch.tensor([1 for n in num_patches for i in range(n)], dtype=torch.long),
        )
        return ret, images

    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]], is_train = True) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        self.is_train = is_train
        
        total_completion_mask = torch.zeros(len(inputs), self.max_completion_length, device=device)
        total_completion_mask_phase1 = torch.zeros(len(inputs), self.max_completion_length, device=device)
        total_completion_mask_phase2 = torch.zeros(len(inputs), self.max_completion_length, device=device)
        total_completion_ids = torch.full(
            (len(inputs), self.max_completion_length), self.processing_class.pad_token_id, device=device
        )
        total_completion_pointers = np.zeros((len(inputs),), dtype=int)
        finish_flags = np.zeros((len(inputs),), dtype=bool)
        phase1_texts = [""] * len(inputs)
        phase2_texts = [""] * len(inputs)
        phase1_parent_count = len(inputs)
        phase2_parent_index = None
        phase2_prompt_index = None
        phase2_child_index = None
        phase2_group_size = 1
        phase1_prompt_count = 0
        original_pixel_values = None
        original_image_grid_thw = None
        
        j=-1
        while True:
            j+=1
            
            if j == 0:
                original_prompts = [x["message"] for x in inputs]
                inputs_tobe_updated_each_turn = inputs.copy()
            if "qwen" in self.model_id.lower(): 
                text = [self.processing_class.apply_chat_template(
                    inp['message'], 
                    tokenize=False, 
                    add_generation_prompt=True
                ) for inp in inputs_tobe_updated_each_turn]
                image_inputs, video_inputs = [], []
                            
                image_inputs, video_inputs = [], []
                for inp in inputs_tobe_updated_each_turn:
                    image_input, video_input = process_vision_info(inp['message'])
                    if image_input:
                        image_inputs.append(image_input)
                    if video_input:
                        video_inputs.append(video_input)
                
                for i in range(len(text)):
                    if "qwen3" not in self.model_id.lower() and len(text[i].split('<|im_end|>\n<|im_start|>assistant\n')) > 2:
                        t_list = text[i].split('<|im_end|>\n<|im_start|>assistant\n')[:-1]
                        text[i] = '<|im_end|>\n<|im_start|>assistant\n'.join(t_list[:2]) + ''.join(t_list[2:])
                
                video_list = video_inputs if len(video_inputs) > 0 else None
                image_list = image_inputs if len(image_inputs) > 0 else None
                prompt_inputs = self.processing_class(
                    text=text,
                    images=image_list,
                    videos=video_list,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt"
                )
            elif "internvl" in self.model_id.lower():
                prompt_inputs, image_inputs = self.multi_modal_get_item(inputs_tobe_updated_each_turn)

            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            if "qwen" in self.model_id.lower():
                self._align_qwen_vl_input_ids_attention_mask(prompt_inputs)
            latest_prompt_ids, latest_prompt_mask, latest_pixel_values = \
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"], prompt_inputs.get("pixel_values")
            
            if latest_pixel_values is not None:
                model_dtype = next(self.model.parameters()).dtype
                if latest_pixel_values.dtype != model_dtype:
                    prompt_inputs["pixel_values"] = latest_pixel_values.to(model_dtype)
                    latest_pixel_values = prompt_inputs["pixel_values"]

            latest_image_grid_thw = prompt_inputs.get("image_grid_thw")

            gen_model_inputs = None
            if "qwen" in self.model_id.lower():
                gen_model_inputs = self._qwen_vl_generate_inputs(prompt_inputs)
                latest_prompt_ids = gen_model_inputs["input_ids"]
                if "attention_mask" in gen_model_inputs:
                    latest_prompt_mask = gen_model_inputs["attention_mask"]
                else:
                    pad_id = getattr(self.processing_class.tokenizer, "pad_token_id", None)
                    if pad_id is not None:
                        latest_prompt_mask = (latest_prompt_ids != pad_id).long()
                    else:
                        latest_prompt_mask = torch.ones_like(latest_prompt_ids, dtype=torch.long, device=latest_prompt_ids.device)
            
            if self.max_turns == j:
                break
            

            if j == 0:
                if latest_prompt_ids.size(1) > self.max_prompt_length:
                    print(f"Warning: prompt length {latest_prompt_ids.size(1)} exceeds the maximum prompt length {self.max_prompt_length}. Truncating.")
                original_prompt_ids = latest_prompt_ids.clone()
                original_prompt_mask = latest_prompt_mask.clone()
                grounded_images = []
                if "qwen" in self.model_id.lower():
                    if latest_pixel_values is not None:
                        original_pixel_values = latest_pixel_values.clone()
                    if latest_image_grid_thw is not None:
                        original_image_grid_thw = latest_image_grid_thw.clone()

            turn_generation_config = deepcopy(self.generation_config)
            if self.graph_two_phase:
                turn_generation_config.stop_strings = ["</graph>"] if j == 0 else ["</answer>"]
            else:
                if turn_generation_config.stop_strings is None:
                    turn_generation_config.stop_strings = ["</answer>"]
                elif "</answer>" not in turn_generation_config.stop_strings:
                    turn_generation_config.stop_strings += ["</answer>"]
                
            p_mask = latest_prompt_mask
                
            start_time = time.time()
            if "qwen" in self.model_id.lower():
                with torch.no_grad():
                    qwen_generate_kwargs = {}
                    if "qwen3" in self.model_id.lower():
                        qwen_generate_kwargs["use_cache"] = True
                    qwen_generate_kwargs["tokenizer"] = self.processing_class.tokenizer
                    assert gen_model_inputs is not None
                    base_model = self.accelerator.unwrap_model(self.model)
                    gc_disabled = False
                    if "qwen3" in self.model_id.lower() and hasattr(base_model, "is_gradient_checkpointing"):
                        if getattr(base_model, "is_gradient_checkpointing", False) and hasattr(base_model, "gradient_checkpointing_disable"):
                            base_model.gradient_checkpointing_disable()
                            gc_disabled = True

                    try:
                        if self.is_train:
                            
                            prompt_completion_ids = self.model.generate(
                                **gen_model_inputs, generation_config=turn_generation_config, **qwen_generate_kwargs
                            )
                        else:
                            eval_generation_config = deepcopy(turn_generation_config)
                            eval_generation_config.temperature = 0.001
                            eval_generation_config.do_sample = True
                            eval_generation_config.top_k = 1
                            eval_generation_config.top_p = 0.
                            prompt_completion_ids = self.model.generate(
                                    **gen_model_inputs,
                                    generation_config=eval_generation_config, **qwen_generate_kwargs
                                )
                    finally:
                        if gc_disabled and hasattr(base_model, "gradient_checkpointing_enable"):
                            base_model.gradient_checkpointing_enable()
                        
                prompt_length = latest_prompt_ids.size(1)
                if not (latest_prompt_ids == prompt_completion_ids[:, :prompt_length]).all():
                    print("Prompt mismatch, unexpected.")
                completion_ids = prompt_completion_ids[:, prompt_length:]
                
            elif "internvl" in self.model_id.lower():
                with torch.no_grad():
                    if self.is_train:
                        
                        prompt_completion_ids = self.model.generate(
                                input_ids=prompt_inputs["input_ids"], 
                                attention_mask=prompt_inputs["attention_mask"], 
                                pixel_values=prompt_inputs.get("pixel_values"), 
                                generation_config=turn_generation_config, 
                                tokenizer=self.tokenizer
                            )
                    else:
                        eval_generation_config = deepcopy(turn_generation_config)
                        eval_generation_config.temperature = 0.001
                        eval_generation_config.do_sample = True
                        eval_generation_config.top_k = 1
                        eval_generation_config.top_p = 0.
                        prompt_completion_ids = self.model.generate(
                                input_ids=prompt_inputs["input_ids"], 
                                attention_mask=prompt_inputs["attention_mask"], 
                                pixel_values=prompt_inputs.get("pixel_values"), 
                                generation_config=eval_generation_config, 
                                tokenizer=self.tokenizer
                            )

                completion_ids = prompt_completion_ids

            time_taken = time.time() - start_time
            print(f"\n[Timer]: \nTime taken for generation: {time_taken:.2f} seconds.\n")

    

            start_time = time.time()

            is_eos = completion_ids == self.eos_token_id
            is_pad = completion_ids == self.pad_token_id
            is_eos = is_eos | is_pad
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            
            
            
            for i in range(len(completion_ids)):
                if total_completion_pointers[i] < self.max_completion_length and not finish_flags[i]:
                    left_tk_space = min(self.max_completion_length, total_completion_pointers[i] + eos_idx[i]) - total_completion_pointers[i]
                    start_ptr = total_completion_pointers[i]
                    end_ptr = total_completion_pointers[i] + left_tk_space
                    total_completion_ids[i, start_ptr:end_ptr] = completion_ids[i, :left_tk_space]
                    total_completion_mask[i, start_ptr:end_ptr] = completion_mask[i, :left_tk_space]
                    if self.graph_two_phase and j == 0:
                        total_completion_mask_phase1[i, start_ptr:end_ptr] = completion_mask[i, :left_tk_space]
                    elif self.graph_two_phase and j == 1:
                        total_completion_mask_phase2[i, start_ptr:end_ptr] = completion_mask[i, :left_tk_space]
                    total_completion_pointers[i] += eos_idx[i]
                    completion = self.processing_class.decode(completion_ids[i][:eos_idx[i]], skip_special_tokens=True)
                    if self.graph_two_phase and j == 0:
                        phase1_texts[i] = completion
                    elif self.graph_two_phase and j == 1:
                        phase2_texts[i] = completion
                    if 'assistant' != inputs_tobe_updated_each_turn[i]['message'][-1]['role']:
                        inputs_tobe_updated_each_turn[i]['message'].append({'role': 'assistant', 'content': []})
                    inputs_tobe_updated_each_turn[i]['message'][-1]['content'] += [{"type": "text", "text": completion}]
                    print(i, '_', j, completion)
                    image_with_bbox = None
                    response = ''
                    if self.graph_two_phase:
                        if j == 0:
                            graph_json = self._extract_graph_json(completion)
                            question_text = inputs[i].get("question", "")
                            use_image_in_phase2 = (
                                (not self.is_train)
                                or (self.state.global_step >= self.phase2_no_image_steps)
                            )
                            image_path = inputs[i]["image"] if use_image_in_phase2 else None
                            turn2_user = self._build_turn2_user_message(question_text, graph_json, image_path=image_path)
                            inputs_tobe_updated_each_turn[i]["message"] = [turn2_user]
                        else:
                            finish_flags[i] = True
                    else:
                        tool_call_query = self.parse_grounding(completion)
                        if tool_call_query is None:
                            finish_flags[i] = True
                        else:
                            if self.args.log_completions and 'eval' in inputs[i] and inputs[i]['eval']:
                                buffered = io.BytesIO()
                                image_inputs[i][-1].save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                try:
                                    image_base64, response = self.tools[self.tool_func](img_str = img_str, tool_call_query = tool_call_query, normalized_bboxs = True if 'internvl' in self.model_id else False)
                                    image_with_bbox = base64.b64decode(image_base64)
                                    image_with_bbox = Image.open(io.BytesIO(image_with_bbox))
                                    if 'crop' in self.args.setting:
                                        bbox = re.search(r'\d+, \d+, \d+, \d+', response)[0]
                                        bbox = [int(x) for x in bbox.split(', ')]
                                        image_with_bbox = image_with_bbox.crop(bbox)
                                except Exception as error:
                                    if response == '':
                                        response = f"Failure: {str(error)}"


                    temp_image_list = None
                    if image_with_bbox:
                        grounded_images.append(image_with_bbox)
                    else:
                        grounded_images.append(None)

                                          
                    print("tool response: ",response)

            if self.graph_two_phase and j == 0:
                phase1_prompt_count = max(1, phase1_parent_count // max(1, self.num_generations))
                phase2_group_size = max(1, self.phase2_num_generations)
                parent_count = len(inputs_tobe_updated_each_turn)
                expanded_inputs = []
                expanded_phase1_texts = []
                parent_indices = []
                prompt_indices = []
                child_indices = []
                for parent_idx in range(parent_count):
                    prompt_idx = parent_idx // max(1, self.num_generations)
                    for child_idx in range(phase2_group_size):
                        expanded_inputs.append(deepcopy(inputs_tobe_updated_each_turn[parent_idx]))
                        expanded_phase1_texts.append(phase1_texts[parent_idx])
                        parent_indices.append(parent_idx)
                        prompt_indices.append(prompt_idx)
                        child_indices.append(child_idx)
                inputs_tobe_updated_each_turn = expanded_inputs
                inputs = expanded_inputs
                phase1_texts = expanded_phase1_texts
                phase2_texts = [""] * len(expanded_inputs)
                phase2_parent_index = torch.tensor(parent_indices, dtype=torch.long, device=device)
                phase2_prompt_index = torch.tensor(prompt_indices, dtype=torch.long, device=device)
                phase2_child_index = torch.tensor(child_indices, dtype=torch.long, device=device)
                idx_expand = phase2_parent_index
                total_completion_ids = total_completion_ids.index_select(0, idx_expand)
                total_completion_mask = total_completion_mask.index_select(0, idx_expand)
                total_completion_mask_phase1 = total_completion_mask_phase1.index_select(0, idx_expand)
                total_completion_mask_phase2 = total_completion_mask_phase2.index_select(0, idx_expand)
                total_completion_pointers = total_completion_pointers[parent_indices]
                finish_flags = np.array([False] * len(expanded_inputs), dtype=bool)
                original_prompt_ids = original_prompt_ids.index_select(0, idx_expand)
                original_prompt_mask = original_prompt_mask.index_select(0, idx_expand)
                original_prompts = [deepcopy(original_prompts[p]) for p in parent_indices]
                if original_pixel_values is not None:
                    original_pixel_values = None
                if original_image_grid_thw is not None:
                    original_image_grid_thw = None
                expected_size = parent_count * phase2_group_size
                if len(inputs_tobe_updated_each_turn) != expected_size:
                    raise RuntimeError(
                        f"Phase2 expansion mismatch: got {len(inputs_tobe_updated_each_turn)} rows, expected {expected_size}."
                    )
                local_group_counts = torch.bincount(phase2_parent_index, minlength=parent_count)
                if not torch.all(local_group_counts == phase2_group_size):
                    raise RuntimeError(
                        f"Phase2 parent-child mapping mismatch: expected each parent to have {phase2_group_size} children, "
                        f"got counts={local_group_counts.tolist()}."
                    )

        for i in range(len(completion_ids)):
            if total_completion_pointers[i] < self.max_completion_length:
                total_completion_ids[i, total_completion_pointers[i]] = self.eos_token_id
                total_completion_mask[i, total_completion_pointers[i]] = 1
                total_completion_pointers[i] += 1

        completion_ids = total_completion_ids
        completion_mask = total_completion_mask
        
        is_eos = completion_ids == self.eos_token_id
        is_pad = completion_ids == self.pad_token_id
        is_eos = is_eos | is_pad
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_no_tool = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([original_prompt_mask, completion_mask_no_tool], dim=1)
        prompt_completion_ids = torch.cat([original_prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        if is_train:
            with torch.inference_mode():
                num_iterations = getattr(self.args, "num_iterations", 1)
                steps_per_generation = getattr(self.args, "steps_per_generation", 1)
                grad_accum_steps = getattr(self.args, "gradient_accumulation_steps", 1)
                if num_iterations > 1 or steps_per_generation > grad_accum_steps:
                    logp_pixel = (
                        original_pixel_values if original_pixel_values is not None else latest_pixel_values
                    )
                    logp_grid = (
                        original_image_grid_thw if original_image_grid_thw is not None else latest_image_grid_thw
                    )
                    old_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logp_pixel,
                        logp_grid,
                        logits_to_keep,
                    )
                else:
                    old_per_token_logps = None
        else:
            old_per_token_logps = None

        solver_ppl = None
        if self.graph_two_phase:
            with torch.inference_mode():
                logp_pixel = original_pixel_values if original_pixel_values is not None else latest_pixel_values
                logp_grid = original_image_grid_thw if original_image_grid_thw is not None else latest_image_grid_thw
                solver_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logp_pixel,
                    logp_grid,
                    logits_to_keep,
                )
                phase2_mask = total_completion_mask_phase2.float()
                phase2_lengths = phase2_mask.sum(dim=1).clamp_min(1.0)
                phase2_mean_nll = -(solver_per_token_logps * phase2_mask).sum(dim=1) / phase2_lengths
                solver_ppl = torch.exp(phase2_mean_nll).detach()
                 
        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for tool call and logit probability computation: {time_taken:.2f} seconds.\n")

        
        start_time = time.time()
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(original_prompts), len(self.reward_funcs), device=device)
        solver_gpt_scores = None
        phase1_breakdown = {}
        phase2_breakdown = {}

        solver_ppl_for_reward = None
        if solver_ppl is not None:
            if self.graph_two_phase and phase2_parent_index is not None:
                ppl_cpu = solver_ppl.detach().cpu().numpy().astype(np.float64)
                parent_sums = np.zeros((phase1_parent_count,), dtype=np.float64)
                parent_counts = np.zeros((phase1_parent_count,), dtype=np.float64)
                parent_idx_cpu = phase2_parent_index.detach().cpu().numpy().astype(np.int64)
                for row_idx, parent_idx in enumerate(parent_idx_cpu):
                    if parent_idx < 0 or parent_idx >= phase1_parent_count:
                        continue
                    parent_sums[parent_idx] += float(ppl_cpu[row_idx])
                    parent_counts[parent_idx] += 1.0
                parent_mean_ppl = np.divide(
                    parent_sums,
                    np.maximum(parent_counts, 1.0),
                )
                solver_ppl_for_reward = [
                    float(parent_mean_ppl[parent_idx_cpu[row_idx]])
                    for row_idx in range(len(parent_idx_cpu))
                ]
            else:
                solver_ppl_for_reward = solver_ppl.detach().cpu().tolist()

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):  
            reward_start_time = time.time()
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(original_prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(original_prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                exclude = frozenset(["message", "completion"])
                keys_sets = [set(example.keys()) - exclude for example in inputs]
                keys = list(set.intersection(*keys_sets)) if keys_sets else []
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                if "internvl" in self.model_id.lower():
                    reward_kwargs["normalized_bboxs"] = True
                if self.graph_two_phase:
                    reward_kwargs["phase1_completion"] = phase1_texts
                    reward_kwargs["phase2_completion"] = phase2_texts
                    if phase2_parent_index is not None:
                        reward_kwargs["phase2_parent_index"] = phase2_parent_index.detach().cpu().tolist()
                    if phase2_prompt_index is not None:
                        reward_kwargs["phase2_prompt_index"] = phase2_prompt_index.detach().cpu().tolist()
                    if phase2_child_index is not None:
                        reward_kwargs["phase2_child_index"] = phase2_child_index.detach().cpu().tolist()
                    correctness_scores_container = {}
                    reward_kwargs["_correctness_scores_container"] = correctness_scores_container
                    reward_kwargs["_phase1_breakdown_container"] = phase1_breakdown
                    reward_kwargs["_phase2_breakdown_container"] = phase2_breakdown
                    if solver_gpt_scores is not None:
                        reward_kwargs["solver_gpt_scores"] = solver_gpt_scores
                    if solver_ppl_for_reward is not None:
                        reward_kwargs["solver_ppl"] = solver_ppl_for_reward
                output_reward_func = reward_func(prompts=original_prompts, completions=completions, completion_ids = completion_ids, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                if self.graph_two_phase and getattr(reward_func, "__name__", "") == "phase2_combined_reward":
                    solver_gpt_scores = correctness_scores_container.get("scores", None)
                    if solver_gpt_scores is not None and phase2_parent_index is not None:
                        parent_scores = np.zeros((phase1_parent_count,), dtype=np.float32)
                        parent_counts = np.zeros((phase1_parent_count,), dtype=np.float32)
                        parent_idx_cpu = phase2_parent_index.detach().cpu().numpy().astype(np.int64)
                        for row_idx, parent_idx in enumerate(parent_idx_cpu):
                            if parent_idx < 0 or parent_idx >= phase1_parent_count:
                                continue
                            parent_scores[parent_idx] += float(solver_gpt_scores[row_idx])
                            parent_counts[parent_idx] += 1.0
                        parent_mean = np.divide(
                            parent_scores,
                            np.maximum(parent_counts, 1.0),
                        )
                        solver_gpt_scores = [float(parent_mean[parent_idx_cpu[row_idx]]) for row_idx in range(len(parent_idx_cpu))]
            
            reward_time = time.time() - reward_start_time
            print(f"\n[Timer]: \nTime taken for reward function {reward_func.__name__}: {reward_time:.2f} seconds.\n")

        current_batch_rewards = rewards_per_func.clone()
        rewards_per_func = gather(rewards_per_func)
        
        
        idx_phase1_reward = None
        idx_phase2_reward = None
        
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            if "phase1_combined_reward" in reward_func_name:
                idx_phase1_reward = i
            elif "phase2_combined_reward" in reward_func_name:
                idx_phase2_reward = i

        rewards = rewards_per_func.sum(dim=1)

        if not is_train:
            advantages = torch.zeros_like(rewards)
            advantages_phase1 = torch.zeros_like(rewards)
            advantages_phase2 = torch.zeros_like(rewards)
            std_grouped_rewards = rewards
        elif self.graph_two_phase and idx_phase2_reward is not None and idx_phase1_reward is not None:
            phase1_rewards = rewards_per_func[:, idx_phase1_reward]
            phase2_rewards = rewards_per_func[:, idx_phase2_reward]
            if phase2_parent_index is not None:
                phase2_parent_global_local = phase2_parent_index + (self.accelerator.process_index * phase1_parent_count)
                parent_global_ids = gather(phase2_parent_global_local)
                unique_parent_ids = torch.unique(parent_global_ids, sorted=True)
                mean_grouped_rewards = torch.zeros_like(phase2_rewards)
                std_grouped_rewards = torch.zeros_like(phase2_rewards)
                for pid in unique_parent_ids:
                    mask = parent_global_ids == pid
                    vals = phase2_rewards[mask]
                    mean_grouped_rewards[mask] = vals.mean()
                    std_grouped_rewards[mask] = vals.std()

                if phase2_prompt_index is not None:
                    phase2_prompt_global_local = phase2_prompt_index + (self.accelerator.process_index * phase1_prompt_count)
                    prompt_global_ids = gather(phase2_prompt_global_local)
                    unique_prompt_ids = torch.unique(prompt_global_ids, sorted=True)
                    mean_grouped_phase1 = torch.zeros_like(phase1_rewards)
                    std_grouped_phase1 = torch.zeros_like(phase1_rewards)
                    for pgid in unique_prompt_ids:
                        mask = prompt_global_ids == pgid
                        vals = phase1_rewards[mask]
                        mean_grouped_phase1[mask] = vals.mean()
                        std_grouped_phase1[mask] = vals.std()
                else:
                    mean_grouped_phase1 = phase1_rewards
                    std_grouped_phase1 = torch.zeros_like(phase1_rewards)
            else:
                mean_grouped_rewards = phase2_rewards.view(-1, self.num_generations).mean(dim=1)
                std_grouped_rewards = phase2_rewards.view(-1, self.num_generations).std(dim=1)
                mean_grouped_phase1 = phase1_rewards.view(-1, self.num_generations).mean(dim=1)
                std_grouped_phase1 = phase1_rewards.view(-1, self.num_generations).std(dim=1)
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                mean_grouped_phase1 = mean_grouped_phase1.repeat_interleave(self.num_generations, dim=0)
                std_grouped_phase1 = std_grouped_phase1.repeat_interleave(self.num_generations, dim=0)
            advantages_phase1 = (phase1_rewards - mean_grouped_phase1) / (std_grouped_phase1 + 1e-4)
            advantages_phase2 = (phase2_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            advantages = advantages_phase2
        else:
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            advantages_phase1 = advantages
            advantages_phase2 = advantages

        process_slice = slice(
            self.accelerator.process_index * len(original_prompts),
            (self.accelerator.process_index + 1) * len(original_prompts),
        )
        advantages = advantages[process_slice]
        advantages_phase1 = advantages_phase1[process_slice]
        advantages_phase2 = advantages_phase2[process_slice]
        

        local_dataset_names = [example.get("dataset", "unknown_dataset") for example in inputs]
        gathered_dataset_names = gather_object(local_dataset_names)
        if (
            isinstance(gathered_dataset_names, list)
            and len(gathered_dataset_names) > 0
            and isinstance(gathered_dataset_names[0], list)
        ):
            global_dataset_names = [name for per_rank in gathered_dataset_names for name in per_rank]
        else:
            global_dataset_names = list(gathered_dataset_names)

        if len(global_dataset_names) != rewards_per_func.shape[0]:
            fallback_name = local_dataset_names[0] if len(local_dataset_names) > 0 else "unknown_dataset"
            global_dataset_names = [fallback_name] * rewards_per_func.shape[0]

        all_dataset_key = "all_dataset"
        unique_dataset_names = sorted(set(global_dataset_names))
        dataset_masks = {
            dataset_name: torch.tensor(
                [name == dataset_name for name in global_dataset_names],
                dtype=torch.bool,
                device=device,
            )
            for dataset_name in unique_dataset_names
        }

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            if self.graph_two_phase and reward_func_name in (
                "phase1_combined_reward",
                "phase2_combined_reward",
            ):
                continue
            self._metrics[f"{all_dataset_key}/rewards/{reward_func_name}"].append(rewards_per_func[:, i].mean().item())
            for dataset_name, mask in dataset_masks.items():
                if mask.any():
                    dataset_reward_mean = rewards_per_func[mask, i].mean().item()
                    self._metrics[f"{dataset_name}/rewards/{reward_func_name}"].append(dataset_reward_mean)

        self._metrics[f"{all_dataset_key}/reward"].append(rewards.mean().item())
        self._metrics[f"{all_dataset_key}/reward_std"].append(std_grouped_rewards.mean().item())
        for dataset_name, mask in dataset_masks.items():
            if not mask.any():
                continue
            self._metrics[f"{dataset_name}/reward"].append(rewards[mask].mean().item())
            self._metrics[f"{dataset_name}/reward_std"].append(std_grouped_rewards[mask].mean().item())
        if self.graph_two_phase:
            phase1_metric_map = {
                "r_json": "r_json",
                "r_structure": "r_structure",
                "r_node": "r_node",
                "r_leak_penalty": "r_leak_penalty",
                "r_perplexity": "r_perplexity",
                "r_format_total": "r_format_total",
                "phase1_total": "total",
            }
            phase2_metric_map = {
                "gptscore": "gptscore",
                "r_json": "r_json",
                "r_structure": "r_structure",
                "r_operations": "r_operations",
                "r_node": "r_node",
                "r_format_total": "format_total",
                "phase2_total": "total",
            }

            for key, metric_key in phase1_metric_map.items():
                values = phase1_breakdown.get(key)
                if values is None:
                    continue
                values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
                values_tensor = gather(values_tensor)
                self._metrics[f"{all_dataset_key}/rewards_phase1/{metric_key}"].append(values_tensor.mean().item())
                for dataset_name, mask in dataset_masks.items():
                    if mask.any():
                        self._metrics[f"{dataset_name}/rewards_phase1/{metric_key}"].append(values_tensor[mask].mean().item())

            for key, metric_key in phase2_metric_map.items():
                values = phase2_breakdown.get(key)
                if values is None:
                    continue
                values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
                values_tensor = gather(values_tensor)
                self._metrics[f"{all_dataset_key}/rewards_phase2/{metric_key}"].append(values_tensor.mean().item())
                for dataset_name, mask in dataset_masks.items():
                    if mask.any():
                        self._metrics[f"{dataset_name}/rewards_phase2/{metric_key}"].append(values_tensor[mask].mean().item())
            if idx_phase1_reward is not None and idx_phase2_reward is not None:
                phase_total = rewards_per_func[:, idx_phase1_reward] + rewards_per_func[:, idx_phase2_reward]
                self._metrics[f"{all_dataset_key}/rewards/phase_total"].append(phase_total.mean().item())
                for dataset_name, mask in dataset_masks.items():
                    if mask.any():
                        self._metrics[f"{dataset_name}/rewards/phase_total"].append(phase_total[mask].mean().item())

        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for reward computation: {time_taken:.2f} seconds.\n")


        return_content = {
            "step": str(self.state.global_step),
            "input_output_text": inputs_tobe_updated_each_turn,
            "reward_name": [reward_func_name.__name__ for reward_func_name in self.reward_funcs],
            "reward_list": current_batch_rewards.tolist(),
            "raw_inputs": inputs,
            "image_inputs": image_inputs,
            "grounded_images": grounded_images,
            "prompt_ids": original_prompt_ids,
            "prompt_mask": original_prompt_mask,
            "pixel_values": (
                original_pixel_values if original_pixel_values is not None else latest_pixel_values
            ),
            "image_grid_thw": (
                original_image_grid_thw if original_image_grid_thw is not None else latest_image_grid_thw
            ),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completion_mask_phase1": total_completion_mask_phase1,
            "completion_mask_phase2": total_completion_mask_phase2,
            "old_per_token_logps": old_per_token_logps,
            "advantages": advantages,
            "advantages_phase1": advantages_phase1,
            "advantages_phase2": advantages_phase2,
        }
    
        return return_content


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        start_time = time.time()
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        pixel_values, image_grid_thw = inputs['pixel_values'], inputs['image_grid_thw']
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        is_eos = completion_ids == self.processing_class.eos_token_id
        is_pad = completion_ids == self.processing_class.pad_token_id
        is_eos = is_eos | is_pad
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.accelerator.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.accelerator.device).expand(is_eos.size(0), -1)
        completion_mask_no_tool = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask_no_tool], dim=1)
        logits_to_keep = completion_ids.size(1)

        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep
                        )

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep
        )
        if self.beta != 0.0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        advantages_phase1 = inputs.get("advantages_phase1", advantages)
        advantages_phase2 = inputs.get("advantages_phase2", advantages)
        completion_mask_phase1 = inputs.get("completion_mask_phase1")
        completion_mask_phase2 = inputs.get("completion_mask_phase2")
        old_per_token_logps = (
            per_token_logps.detach() if inputs.get("old_per_token_logps") is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - 0.28, 1 + 0.28)
        if self.graph_two_phase and completion_mask_phase1 is not None and completion_mask_phase2 is not None:
            token_advantages = advantages_phase1.unsqueeze(1) * completion_mask_phase1 + advantages_phase2.unsqueeze(1) * completion_mask_phase2
        else:
            token_advantages = advantages.unsqueeze(1)
        per_token_loss1 = coef_1 * token_advantages
        per_token_loss2 = coef_2 * token_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)


        def find_sublist_indices(full_list_of_list, sublist, get_end_id = False):
            sub_len = len(sublist)
            matched_indices = []
            for full_list in full_list_of_list:
                for i in range(len(full_list) - sub_len + 1):
                    if full_list[i:i + sub_len] == sublist:
                        if get_end_id:
                            matched_indices.append(i + sub_len - 1)
                        else:
                            matched_indices.append(i)
                        break
                else:
                    matched_indices.append(9999)
            return matched_indices
        
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        denom = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for loss computation: {time_taken:.2f} seconds.\n")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        
        new_inputs = []
        existed_inputs_image_answer_question = []
        for i in range(len(inputs)):
            inputs_image_answer_question = inputs[i]['image'] + inputs[i]['gt_answer'] + inputs[i]['message'][0]['content'][-1]['text']
            if inputs_image_answer_question not in existed_inputs_image_answer_question:
                existed_inputs_image_answer_question.append(inputs_image_answer_question)
                new_inputs.append(inputs[i])
        inputs = new_inputs
                
        
        for i in range(len(inputs)):
            inputs[i]['eval'] = True
        inputs = self._prepare_inputs(inputs, is_train=False)
        for j in range(len(inputs['input_output_text'])):
            for m in range(len(inputs['input_output_text'][j]['message'])):
                for i in range(len(inputs['input_output_text'][j]['message'][m]['content'])):
                    if 'type' in inputs['input_output_text'][j]['message'][m]['content'][i] and inputs['input_output_text'][j]['message'][m]['content'][i]['type'] == 'image':
                        inputs['input_output_text'][j]['message'][m]['content'][i]['image'] = 'New image with bounding box.'
                
        context_images = []
        for i in range(len(inputs['image_inputs'])):
            image_list = list(inputs['image_inputs'][i]) if isinstance(inputs['image_inputs'][i], list) else [inputs['image_inputs'][i]]
            grounded = inputs['grounded_images'][i] if i < len(inputs['grounded_images']) else None
            if grounded is not None:
                image_list.append(grounded)
            context_images.append(image_list)
        
        local_log_path = self.args.output_dir + f"/local_log_step_{inputs['step']}/evaluation_results_{self.accelerator.process_index}.json"
        local_log_image_path = self.args.output_dir + f"/local_log_step_{inputs['step']}/evaluation_images/"
        os.makedirs(local_log_image_path, exist_ok=True)
            
        existing_logs = []
        if os.path.exists(local_log_path):
            with open(local_log_path, 'r') as f:
                existing_logs = json.load(f)
        
        for i in range(len(inputs['input_output_text'])):
            conversation_images_per_conv = []
            image_name_prefix = inputs['raw_inputs'][i]['image'].split('/')[-1][:10].split('.')[0]  + f"_{str(random.randint(0, 1000))}"
            for j in range(len(context_images[i])):
                output_image_path = os.path.join(local_log_image_path, f"{image_name_prefix}_{j}.png")
                context_images[i][j].save(output_image_path)
                conversation_images_per_conv.append(output_image_path)
            
            existing_logs.append(
                                    {
                                        "input_output_conv":inputs['input_output_text'][i],
                                        "conversation_images": conversation_images_per_conv,
                                        "reward_name": inputs['reward_name'],
                                        "reward_list": str(inputs['reward_list'][i]),
                                    }
                                 )
        json.dump(existing_logs, open(local_log_path, 'w'))
        if self.is_train:
            
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)
                loss = loss.mean().detach()
            return loss, None, None
        else:
            return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
