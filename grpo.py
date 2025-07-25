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
    completion_contents = [completion[0]["content"]
                           for completion in completions]

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
    completion_contents = [completion[0]["content"]
                           for completion in completions]

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
            progress = gen_len / max_len  # 距离最大长度有多远
            cosine = math.cos(progress * math.pi)  # 计算余弦

            if acc_reward > 0.5:  # 如果回答是正确答案
                min_value = min_value_correct
                max_value = max_value_correct
            else:  # 回答不正确
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
        words = text.lower().split()  # 转换成小写然后分割
        return zip(*[words[i:] for i in range(ngram_size)])  # 创建 n-grams

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":  # 对空的回答不进行惩罚
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:  # 对短回答不进行惩罚
                rewards.append(0.0)
                continue

            ngrams = set()  # 使用 set 保存去重后的 n-grams
            total = 0
            for ng in zipngram(completion, ngram_size):  # 生成 n-grams
                ngrams.add(ng)  # 去重
                total += 1  # 计数

            # 计算缩放系数：重复越多，scaling 越大
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty  # 计算惩罚
            rewards.append(reward)
        return rewards
    return get_repetition_penalty_reward

# 奖励函数的参数配置


@dataclass
class GRPOScriptArguments:
    """
    GRPO相关配置，特别是奖励函数的配置
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward for cosine scaling for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for cosine scaling for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward for cosine scaling for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for cosine scaling for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for cosine scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={
            "help": "Maximum (negative) penalty for for repetition penalty reward"},
    )


# Define TrainingArguments from transformers
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,          # Output directory for checkpoints and logs
    overwrite_output_dir=True,
    num_train_epochs=1,             # Total number of training epochs
    per_device_train_batch_size=8,  # Batch size per device during training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    learning_rate=5e-5,            # Initial learning rate for AdamW optimizer
    # Linear warmup over warmup_ratio fraction of training steps
    warmup_ratio=0.1,
    # Apply weight decay to all layers except bias and LayerNorm weights
    weight_decay=0.01,
    logging_steps=10,              # Log every X updates steps
    eval_strategy="steps",    # Evaluate every `eval_steps`
    eval_steps=50,                 # Evaluation and logging steps
    save_strategy="steps",         # Save checkpoint every `save_steps`
    save_steps=50,                 # Save checkpoint every X updates steps
    # Limit the total amount of checkpoints. Deletes the older checkpoints.
    save_total_limit=2,
    dataloader_num_workers=2,      # Number of subprocesses to use for data loading
    seed=42,                       # Random seed for reproducibility
    bf16=True,                     # Use mixed precision BFP16 training
    push_to_hub=False,             # Whether to push the final model to Hugging Face Hub
    gradient_checkpointing=True,   # Enable gradient checkpointing
    report_to="none",              # Reporting to no one
)


@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    model_name_or_path: str = field(
        default=MODEL_NAME, metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "Override the default `torch_dtype` and load the model under this dtype."}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading model and tokenizer."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation to use. 'flash_attention_2' or None"}
    )


# Instantiate configuration objects
script_args = GRPOScriptArguments()
model_args = ModelConfig()

# Utility function to get reward functions based on script arguments


def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,  # Assuming accuracy_reward is defined in previous steps
        "format": format_reward,      # Assuming format_reward is defined in previous steps
        # Assuming reasoning_steps_reward is defined
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(  # Assuming get_cosine_scaled_reward is defined
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(  # Assuming get_repetition_penalty_reward is defined
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(
                f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list


logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """
    A simple callback for logging training information at specific steps.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            logger.info(
                f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}")


def get_callbacks(training_args, model_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()]  # Instantiate our LoggingCallback
    return callbacks


# Get reward functions and callbacks
reward_functions = get_reward_functions(script_args)
callbacks = get_callbacks(training_args, model_args, script_args)

# Create GRPOConfig from TrainingArguments
grpo_config = GRPOConfig(
    **training_args.to_dict(),  # Convert TrainingArguments to dictionary and unpack
    **{
        # REMOVED model_init_kwargs here
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
    }
)

grpo_trainer = GRPOTrainer(
    model=model,                      # Our initialized Qwen model
    reward_funcs=reward_functions,    # List of reward functions from previous step
    # GRPOConfig (created from TrainingArguments)
    args=grpo_config,
    train_dataset=dataset['train'],   # Training dataset
    eval_dataset=dataset['test'],    # Evaluation dataset
    callbacks=callbacks              # List of callbacks
)

# Start the GRPO Training Loop
train_result = grpo_trainer.train()

# Define the path to your trained model (same as OUTPUT_DIR)
TRAINED_MODEL_PATH = "data/Qwen-GRPO-training"

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

# Save the trained model
grpo_trainer.save_model(TRAINED_MODEL_PATH)

print(f"GRPO Trained model saved to {TRAINED_MODEL_PATH}")

# Load the tokenizer - make sure to use trust_remote_code=True if needed
tokenizer = AutoTokenizer.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model config requires it
    padding_side="right" # Ensure consistent padding side
)

# Set pad token if it wasn't saved or loaded correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the trained model itself
trained_model = AutoModelForCausalLM.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model architecture requires it
    torch_dtype=torch.bfloat16 # Keep the same dtype as training for consistency
)

# Move the loaded model to your device (GPU if available)
trained_model.to(device) # 'device' is still our CUDA device from before

# Testing Inference with the Trained Model
def test_trained_model_inference(user_input: str):
    """Test inference with the loaded trained model and tokenizer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # Re-use our system prompt
        {"role": "user", "content": user_input}
    ]

    # Apply chat template using our tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate output using our *trained_model*
    outputs = trained_model.generate(
        **inputs,
        max_new_tokens=200, # Maybe generate a bit longer now
        do_sample=True,
        temperature=0.7
    )

    # Decode the generated tokens back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

