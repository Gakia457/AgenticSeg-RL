# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int, use_resize: bool) -> ImageObject:
    if use_resize:
        image = image.resize((840, 840), Image.Resampling.BICUBIC)
    
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
        remove_lisa=False,
        is_stage2=False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        self.is_stage2 = is_stage2
        self.use_resize = False
        if data_path == 'detection-datasets/coco':
            raw_dataset = load_dataset(data_path, split="train")
            # raw_dataset = raw_dataset.filter(lambda x: len(x['objects']['bbox']) >= 3)
            self.use_resize = True
        else:
            # --- [AgenticRL] 支持加载多个数据集目录 ---
            from datasets import concatenate_datasets
            if isinstance(data_path, str):
                path_list = [p.strip() for p in data_path.split(',') if p.strip()]
            elif isinstance(data_path, (list, tuple)):
                path_list = data_path
            else:
                path_list = [data_path]

            if len(path_list) > 1:
                print(f"[AgenticRL] Concatenating {len(path_list)} datasets: {path_list}")
                datasets_to_concat = []
                for p in path_list:
                    datasets_to_concat.append(load_from_disk(p)['train'])
                raw_dataset = concatenate_datasets(datasets_to_concat)
            else:
                raw_dataset = load_from_disk(path_list[0])['train']
            # ------------------------------------------

        
        if remove_lisa:
            raw_dataset = raw_dataset.filter(lambda x: not x['id'].startswith('lisa_plus'))
        self.dataset = raw_dataset
        
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            self.is_qwen3 = True
        else:
            self.is_qwen3 = False
        print("is_qwen3:", self.is_qwen3)
        
        # 让模型输出思维链 + 框 点 标签格式的奖励，分数如下：
        if is_stage2:
            self.user_prompt = "<image>\n" \
                "Please find \"{Question}\" with bbox(es) and point(s). " \
                "Also provide a short label for each object. " \
                "First, understand and summarize what the query —\"{Question}\"— is likely referring to (which object or concept). " \
                "Then apply this to the image and find the matched target object(s). " \
                "Return ALL matching instances; double-check none are missed. " \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
                "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
                "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
                "<answer>{Answer}</answer>"
        else: # 原来的 Stage 1 Prompt 现在stage 1 视为我们的 task1 与其他 task 是并列的
            self.user_prompt = "<image>\n" \
                "Please find \"{Question}\" with bbox(es) and point(s). " \
                "Also provide a short label for each object. " \
                "Compare the difference between object(s) and find the most closely matched object(s). " \
                "Return ALL matching instances; double-check none are missed. " \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
                "Output the bbox(es) and point(s) inside the interested object(s), along with a short label, in JSON format. " \
                "i.e., <think> thinking process (step-by-step reasoning) here </think> " \
                "<answer>{Answer}</answer>"
            
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]
        
        # --- [AgenticRL] 动态推断任务并切换 Prompt ---
        from verl.agentic.router import infer_task_type
        task_type = infer_task_type(row_dict.get("solution"))
        
        current_user_prompt = self.user_prompt
        # Default Task 1 Answer
        example_answer = "[{\"label\": \"chair\", \"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"label\": \"train track\", \"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"

        if task_type == "task2":
            # --- [方案 B: 质量评分 Prompt (Soft Reward)] ---
            current_user_prompt = "<image>\n" \
                "Please evaluate the quality of the current mask for \"{Question}\". " \
                "Provide a quality score between 0.0 (poor quality, needs significant refinement) and 1.0 (good enough, no refinement needed). " \
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
                "Output a JSON object with a \"quality_score\" key. " \
                "The value of \"quality_score\" must be a numeric score, not text. " \
                "i.e., <think> thinking process (step-by-step reasoning) here </think> <answer>{Answer}</answer>"
            example_answer = "{\"quality_score\": score_between_0_and_1}"
            
            # --- [方案 A: 旧的分类 Prompt (Hard Match)] ---
            # current_user_prompt = "<image>\n" \
            #     "Please determine if the current mask for \"{Question}\" is \"good_enough\" or \"need_refine\". " \
            #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. " \
            #     "Output a JSON object with a \"label\" key. " \
            #     "i.e., <think> thinking process (step-by-step reasoning) here </think> <answer>{Answer}</answer>"
            # example_answer = "{\"label\": \"good_enough_or_need_refine\"}"
            # --------------------------------
        elif task_type == "task3":
            # Keep Task 3 prompt formatting aligned with the reward parser contract.
            # [AgenticRL Task3] 2026-05-22b: 放松 <think> 要求，允许只输出 <answer>。
            # 上一版强约束 prompt 保留如下，便于快速回退:
            # current_user_prompt = "<image>\n" \
            #     "Task context: " \
            #     "The green overlay is a mask visualization, not the object's real color. " \
            #     "It is the current mask answer for ALL visual region(s) described by: \"{Question}\". " \
            #     "Your task: " \
            #     "Find the MOST SIGNIFICANT mask error, ignoring minor boundary noise or ambiguous small flaws. " \
            #     "Choose ONE error type: " \
            #     "point_label 0 means EXTRA mask area: green covers a region that should not be in the answer; " \
            #     "point_label 1 means MISSING mask area: a required region, or part of it, should be green but is not covered enough. " \
            #     "Click one clear interior point inside the MOST SIGNIFICANT error region, preferably near the center of that erroneous area rather than on its boundary. " \
            #     "Output rules: " \
            #     "Your response MUST contain exactly two parts: <think>...</think> followed by <answer>...</answer>. " \
            #     "The response MUST start with <think> and MUST NOT contain any text outside these two tags. " \
            #     "In <think>, briefly decide EXTRA or MISSING and identify the main error region. " \
            #     "In <answer>, output ONLY the JSON list, with no explanation and no markdown. " \
            #     "Use exactly this format: <think> brief reasoning </think>" \
            #     "<answer>{Answer}</answer>"
            
            # 05222259  我发现这个 prompt 模型不会输出think 但是性能反而好，有一种野性的直觉的美 
            current_user_prompt = "<image>\n" \
                "Task context: " \
                "The green overlay is a mask visualization, not the object's real color. " \
                "It is the current mask answer for ALL visual region(s) described by: \"{Question}\". " \
                "Your task: " \
                "Find the MOST SIGNIFICANT mask error, ignoring minor boundary noise or ambiguous small flaws. " \
                "Choose ONE error type that best describes the MOST SIGNIFICANT error: " \
                "point_label 0 means EXTRA mask area: green covers a region that should not be in the answer; " \
                "point_label 1 means MISSING mask area: a required region, or part of it, should be green but is not covered enough. " \
                "Do NOT treat green overlay as wrong just because it hides natural colors. " \
                "Click one point inside the MOST SIGNIFICANT error region. " \
                "Output MUST FOLLOW this format: " \
                "<think> EXTRA or MISSING, and where </think> " \
                "<answer>{Answer}</answer>"

            example_answer = "[{\"point_2d\": [x, y], \"point_label\": 0_or_1}]"
            
        # --------------------------------------------

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": current_user_prompt.format(
                Question=row_dict["problem"].lower().strip("."),  # 在这里消费problem字段
                Answer=example_answer
            )},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
            row_dict['image'] = process_image(                    # 消费image字段  注意这里是原地修改row_dict，后续会用到row_dict["images"]来做处理
                row_dict["image"], self.max_pixels, self.min_pixels, self.use_resize
            )
        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["images"] = [
                process_image(image, self.max_pixels, self.min_pixels, self.use_resize) for image in row_dict["images"]
            ]
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "images" in row_dict:          
            # qwen-vl mrope
            if self.is_qwen3:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_5_vl import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row_dict
