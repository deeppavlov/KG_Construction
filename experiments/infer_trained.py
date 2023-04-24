import torch
from peft import PeftModel
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!git clone https://github.com/tloen/alpaca-lora.git
%cd alpaca-lora/
!git checkout 683810b

!pip install -U pip
!pip install -r requirements.txt
!pip install torch==2.0.0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, "finred_checkpoints", torch_dtype=torch.float16)

with open("infer_trained_log.txt", 'a') as out:
    out.write(f"model type before compile: {type(model)} --- model: {model}" + '\n\n')

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model = model.eval()
model = torch.compile(model)

PROMPT_TEMPLATE = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
[INSTRUCTION]
### Input:
[INPUT]
### Response:
"""


def create_prompt(instruction: str, inp: str) -> str:
    return PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction).replace("[INPUT]", inp)


def generate_response(prompt: str, model: PeftModel) -> GreedySearchDecoderOnlyOutput:
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.1,
    )
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
        )


def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
    decoded_output = tokenizer.decode(response.sequences[0])
    response = decoded_output.split("### Response:")[1].strip()
    return "\n".join(textwrap.wrap(response))


def ask_alpaca(instruction: str, inp: str, model: PeftModel = model) -> str:
    prompt = create_prompt(instruction, inp)
    response = generate_response(prompt, model)
    print(format_response(response))


ask_alpaca("Find triplets in the format (subject, relation, object) in the input text.",
           "Lufthansa Passenger Airlines, SWISS and Austrian Airlines are network carriers serving the global market "
           "and all passenger segments.")
