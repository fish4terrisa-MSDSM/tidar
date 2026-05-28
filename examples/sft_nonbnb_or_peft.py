from tidar.utils import load_model_and_tokenizer
from tidar.model import TiDARModel
from tidar.trainer import TiDARSFTTrainer
from trl import SFTConfig
from datasets import load_dataset
import torch


model_name = "/usr/src/ai/models/Olmo2"
base_model, tokenizer = load_model_and_tokenizer(model_name, use_rope_scaling=True)

tidar_model = TiDARModel(base_model, tokenizer, diffusion_block_len=4)

dataset = load_dataset("/usr/src/ai/datasets/pretrain/wikitext-2-raw-v1")

trainer = TiDARSFTTrainer(
    model=tidar_model,
    processing_class=tokenizer,
    train_dataset=dataset['train'],
    args=SFTConfig(
        dataset_text_field="text",
        max_length=1024,
        packing=False,
        dataset_num_proc=8,
        report_to="none",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        warmup_steps=5,
        max_steps=10,
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
trainer.save_model("/usr/src/ai/tmp/tidar-tmp")
