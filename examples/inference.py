from tidar.utils import load_model_and_tokenizer
from tidar.model import TiDARModel
from tidar.generation import TiDARGenerator

base_model, tokenizer = load_model_and_tokenizer("/usr/src/ai/tmp/tidar-tmp", use_rope_scaling=True)
tidar_model = TiDARModel(base_model, tokenizer, diffusion_block_len=4)

generator = TiDARGenerator(tidar_model, draft_len=4)

prompt = "Linux is an operating system that can"

print("--- GENERATING THOUGHTS ---")
thought_prompt = prompt
thought_output = generator.generate(
    prompt=thought_prompt,
    max_new_tokens=200,
    mode="tidar",
    temperature=0.9,
    stop_tokens=[tokenizer.convert_tokens_to_ids("</think>"), tokenizer.eos_token_id],
    log_diffusion=True # Will print the TiDAR drafting logs
)
print(f"Final Stage 1 Output:\n{thought_output}")

regex_pattern = r"^[^vV]+$"
speech_prompt = thought_output
speech_output = generator.generate(
    prompt=speech_prompt,
    max_new_tokens=100,
    temperature=0.9,
    stop_tokens=[tokenizer.convert_tokens_to_ids("</speak>"), tokenizer.eos_token_id],
    regex_pattern=regex_pattern,
    log_diffusion=True
)
print(f"Final Stage 2 Output:\n{speech_output}")
