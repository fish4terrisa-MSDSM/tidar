from tidar.utils import load_model_and_tokenizer, get_bnb_config, get_chat_template
from tidar.model import TiDARModel
from tidar.trainer import TiDARDPOTrainer
from trl import DPOConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch

test_template = """{%- set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{{- '<|im_start|>situation
' + message['content'] -}}{%- if tools is not none -%}{{- '<functions>' -}}{{- tools | tojson -}}{{- '</functions>' -}}{%- elif message.get('functions', none) is not none -%}{{- ' <functions>' + message['functions'] + '</functions>' -}}{%- endif -%}{{- '<|im_end|>
' -}}{%- elif message['role'] == 'user' -%}{{- '<|im_start|>speaker<name>' + [ 'Unknown' ] | random + '</name>
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'assistant' -%}{%- if loop.index0 > 0 -%}{{- '<|im_start|>interaction
' -}}{%- endif -%}{%- if message.get('content', none) is not none -%}{{- '<speak>' + message['content'] + '</speak>' -}}{%- endif -%}{%- if message.get('function_calls', none) is not none -%}{{- '<function_calls>' + message['function_calls'] + '</function_calls>' -}}{% elif message.get('tool_calls', none) is not none %}{{- '<function_calls>' -}}{%- for tool_call in message['tool_calls'] %}{%- if tool_call is mapping and tool_call.get('function', none) is not none %}{%- set args = tool_call['function']['arguments'] -%}{%- set ns = namespace(arguments_list=[]) -%}{%- for key, value in args.items() -%}{%- set ns.arguments_list = ns.arguments_list + [key ~ '=' ~ (value | tojson)] -%}{%- endfor -%}{%- set arguments = ns.arguments_list | join(', ') -%}{{- tool_call['function']['name'] + '(' + arguments + ')' -}}{%- if not loop.last -%}{{ '
' }}{%- endif -%}{% else %}{{- tool_call -}}{%- endif %}{%- endfor %}{{- '</function_calls>' -}}{%- endif -%}{%- if not loop.last -%}{{- '<|im_end|>' + '
' -}}{%- else -%}{{- '<|im_end|>
' -}}{%- endif -%}{%- elif message['role'] == 'environment' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'tool' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- endif -%}{%- if loop.last and add_generation_prompt -%}{{- '<|im_start|>interaction
' -}}{%- endif -%}{%- endfor -%}"""

# Use 4-bit to fit both Policy and Ref models in VRAM
bnb_config = get_bnb_config("4bit")

special_tokens = ["<|mask|>", "<speak>", "</speak>", "<think>", "</think>", "<name>", "</name>", "<functions>", "</functions>", "<function_calls>", "</function_calls>"]

policy_base, tokenizer = load_model_and_tokenizer("/usr/src/ai/tmp/tidar-tmp", bnb_config=bnb_config)

policy_base = prepare_model_for_kbit_training(policy_base)
peft_config = LoraConfig(
    r = 32,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_dropout = 0.05,
    bias = "none",
    trainable_token_indices={'embed_tokens': tokenizer.convert_tokens_to_ids(special_tokens)},
    task_type="CAUSAL_LM"
)
peft_policy_base = get_peft_model(policy_base, peft_config)

policy_model = TiDARModel(peft_policy_base, tokenizer)

tokenizer.eos_token = '<|im_end|>'

tokenizer = get_chat_template(
    tokenizer,
    chat_template = (test_template, tokenizer.eos_token),
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

def formatting_prompts_func(examples, tokenizer):
    prompt       = examples["prompt"]
    chosen      = examples["chosen"]
    rejected    = examples["rejected"]
    final_prompt = []
    final_chosen = []
    final_rejected = []
    for cut_prompt, choice, reject in zip(prompt, chosen, rejected):
        msg_prompt = [{ "role": "user", "content": f"{cut_prompt}" }]
        msg_choice = [{ "role": "assistant", "content": f"{choice}" }]
        msg_reject = [{ "role": "assistant", "content": f"{reject}" }]
        tokenizer.chat_template = test_template
        final_prompt.append(tokenizer.apply_chat_template(msg_prompt, tokenize = False, add_generation_prompt = True))
        tokenizer.chat_template = test_template
        final_chosen.append(tokenizer.apply_chat_template(msg_choice, tokenize = False, add_generation_prompt = False))
        final_rejected.append(tokenizer.apply_chat_template(msg_reject, tokenize = False, add_generation_prompt = False))
    return { "chosen": final_chosen, "rejected": final_rejected, "prompt": final_prompt }
pass

dataset = load_dataset("./strange-dpo-dataset")
dataset = dataset.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, num_proc = 12, batched = True,)

trainer = TiDARDPOTrainer(
    model=policy_model,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    args=DPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        activation_offloading=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        max_steps = 6,
        learning_rate=5e-6,
        bf16=torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps = 1,
        optim_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        report_to = "none",
        output_dir="/usr/src/ai/tmp/outputs-dpo",
        beta=0.1,
        # Internally we use double the length
        max_length=1024,
    ),
)
trainer.train()
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("/usr/src/ai/tmp/tidar-tmp-dpo", save_embedding_layers=True)
tokenizer.save_pretrained("/usr/src/ai/tmp/tidar-tmp-dpo")
