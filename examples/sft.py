from tidar.utils import load_model_and_tokenizer, get_bnb_config
from tidar.model import TiDARModel
from tidar.trainer import TiDARSFTTrainer
from trl import SFTConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch

bnb_config = get_bnb_config("4bit")
special_tokens = ["<|mask|>", "<speak>", "</speak>", "<think>", "</think>", "<name>", "</name>", "<functions>", "</functions>", "<function_calls>", "</function_calls>"]
model_name = "/usr/src/ai/models/Olmo2"
base_model, tokenizer = load_model_and_tokenizer(model_name, use_rope_scaling=True, bnb_config=bnb_config, special_tokens=special_tokens)
base_model = prepare_model_for_kbit_training(base_model)
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
peft_base_model = get_peft_model(base_model, peft_config)
tidar_model = TiDARModel(peft_base_model, tokenizer, diffusion_block_len=4)

dataset = load_dataset("/usr/src/ai/datasets/pretrain/wikitext-2-raw-v1")

trainer = TiDARSFTTrainer(
    model=tidar_model,
    processing_class=tokenizer,
    train_dataset=dataset['train'],
#    peft_config=peft_config,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=2048,
        packing=False,
        dataset_num_proc=8,
        report_to="none",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        warmup_steps=5,
        #max_steps=240,
        num_train_epochs = 1,
        learning_rate=2e-4,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit", # BitsAndBytes Adam
        optim_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        weight_decay=0.01,
        output_dir="/usr/src/ai/tmp/outputs_sft",
    ),
)
trainer.train()
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("/usr/src/ai/tmp/tidar-tmp", save_embedding_layers=True)
tokenizer.save_pretrained("/usr/src/ai/tmp/tidar-tmp")
