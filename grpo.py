# 导入必要的库
import logging
import os
import sys
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional

# 导入PyTorch和Hugging Face Transformers库
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import get_last_checkpoint

# 导入数据集相关库
import datasets
from datasets import load_dataset

# 从 TRL (Transformers Reinforcement Learning) 导入相关库
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    GRPOTrainer,
    GRPOConfig,
    SFTTrainer
)

# 导入数学相关的库
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# 从 DigitalLearningGmbH 加载 "AI-MO/NuminaMath-TIR" 数据集
MATH_le = load_dataset("AI-MO/NuminaMath-TIR", "default")

# 获取训练数据的第一条数据（样本）
print(MATH_le['train'][0])

# Load the "Bespoke-Stratos-17k" dataset from bespokelabs
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default")

# Access the first sample in the training set
print(bespoke_rl['train'][0])

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-GRPO-training"  # 用来保存我们训练的模型

# 创建文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化tokenizer分词器和聊天模板
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)

# 设置pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabulary size: {len(tokenizer)}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")

# 初始化基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

print(f"Model parameters: {model.num_parameters():,}")

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型转移到device
model.to(device)

# 测试基础模型的输出


def test_model_inference(user_input: str):
    """使用加载的模型和分词器测试基础模型的输出。"""
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # 使用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 分词然后生成输出
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# 测试模型
test_input = "how are you?"
response = test_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Model Response: {response}")

# DeepSeek用的系统提示词
SYSTEM_PROMPT = (
    f"""A conversation between User and Assistant. The user asks a question, 
      and the Assistant solves it. The assistant
      first thinks about the reasoning process in the mind and 
      then provides the user with the answer. The reasoning
      process and answer are enclosed within <think> </think> 
      and <answer> </answer> tags, respectively, i.e., 
      <think> reasoning process here </think><answer> answer here </answer>
   """
)


def make_conversation(example):
    """Convert dataset examples into conversation format."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

# 加载和准备数据集
def load_math_dataset():
    """Load and prepare the mathematics dataset."""
    dataset = load_dataset(
        "AI-MO/NuminaMath-TIR",
        name="default",
        split=['train', 'test']
    )
    
    # 转换成字典
    dataset = {
        'train': dataset[0],
        'test': dataset[1]
    }
    
    # 转换格式
    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)

        # 删除`messages`这一列
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    return dataset

dataset = load_math_dataset()

print(f"训练集大小: {len(dataset['train'])}")
print(f"测试集大小: {len(dataset['test'])}")


def validate_dataset(dataset):
    """针对数据集做最基本的检查."""
    
    # 定义数据集需要的字段
    required_fields = ["problem", "prompt"]

    for split in ['train', 'test']:
        print(f"\nValidating {split} split:")

        # 从数据集抽取列的名字
        fields = dataset[split].column_names

        # 检查字段是否丢失
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"Warning: Missing fields: {missing}")
        else:
            print("✓ All required fields present")

        # 抽取第一个样本
        sample = dataset[split][0]

        # 抽取提示词 'prompt' 字段
        messages = sample['prompt']

        # 校验提示词的格式:
        # - 必须包含至少两条信息
        # - 第一条信息必须来自 'system' 角色
        # - 第二条信息必须来自 'user' 角色
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ Prompt format is correct")
        else:
            print("Warning: Incorrect prompt format")

# 验证数据集
validate_dataset(dataset)

def accuracy_reward(completions, solution, **kwargs):
    """
    奖励函数用于检查模型的回答是否在数学上等于标准答案。
    函数使用 latex2sympy2 进行解析，并通过 math_verify 进行验证。
    """
    
    # 抽取响应
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        # 对解答进行解析
        gold_parsed = parse(sol, extraction_mode="first_match", 
                            extraction_config=[LatexExtractionConfig()])
        
        if gold_parsed:  # 如果解析成功
            # 使用宽松的归一化解析模型的答案
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # 如果答案正确，奖励 1.0 , 如果不正确，奖励 0.0 。
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # 如果标准答案无法解析，则赋予中性奖励（0.5）
            reward = 0.5
            print("Warning: Failed to parse gold solution:", sol)

        rewards.append(reward)
    
    return rewards

# 实现格式奖励函数
def format_reward(completions, **kwargs):
  """
  奖励函数会检查补全的回答是否有正确的格式:
  <think>...</think> <answer>...</answer>.
  """
  # 定义正确格式的正则表达式
  pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

  # 从每个回答中抽取内容
  completion_contents = [completion[0]["content"] for completion in completions]

  # 检查每个回答是否符合正则表达式
  matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE)
             for content in completion_contents]

  # 格式正确则奖励 1.0 ，不正确则奖励 0.0
  return [1.0 if match else 0.0 for match in matches]

def reasoning_steps_reward(completions, **kwargs):
    r"""
    奖励函数用于鼓励清晰的逐步推理。
    它会寻找类似“步骤1:”、编号列表符号、未编号列表符号以及过渡词等模式。
    """
    # 匹配推理步骤的正则表达式
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # 抽取回答的内容
    completion_contents = [completion[0]["content"] for completion in completions]

    # 计算每个回答中的推理步骤数量
    matches = [len(re.findall(pattern, content, re.MULTILINE))
               for content in completion_contents]

    # 奖励与推理步骤的数量成正比，最高为1.0。
    # 这里使用了一个“魔法数字”3——鼓励至少完成3个步骤以获得满分奖励。
    return [min(1.0, count / 3) for count in matches]

# 实现余弦缩放奖励函数
def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    返回一个余弦缩放的奖励函数。该函数根据完成长度对准确率奖励进行缩放。
    较短的正确解答获得更高的奖励，较长的错误解答受到较小的惩罚。
    """
    def cosine_scaled_reward(completions, solution, accuracy_rewards, **kwargs):
        """
        余弦缩放的奖励函数，根据回答长度调整准确率奖励。
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
            gen_len = len(content)  # 回答的长度
            progress = gen_len / max_len # 距离最大长度有多远
            cosine = math.cos(progress * math.pi) # 计算余弦

            if acc_reward > 0.5: # 如果回答是正确答案
                min_value = min_value_correct
                max_value = max_value_correct
            else: # 回答不正确
                min_value = max_value_wrong  # 注意这个交换
                max_value = min_value_wrong

            # 余弦缩放公式
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward

def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    返回一个重复惩罚奖励函数。对生成文本中重复出现的n-gram进行惩罚。
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """从文本中生成 n-grams 的帮助函数."""
        words = text.lower().split() # 转换成小写然后分割
        return zip(*[words[i:] for i in range(ngram_size)]) # 创建 n-grams

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "": # 对空的回答不进行惩罚
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size: # 对短回答不进行惩罚
                rewards.append(0.0)
                continue

            ngrams = set() # 使用 set 保存去重后的 n-grams
            total = 0
            for ng in zipngram(completion, ngram_size): # 生成 n-grams
                ngrams.add(ng) # 去重
                total += 1 # 计数

            # 计算缩放系数：重复越多，scaling 越大
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty # 计算惩罚
            rewards.append(reward)
        return rewards
    return get_repetition_penalty_reward

