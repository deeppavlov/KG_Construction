import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "decapoda-research/llama-7b-hf"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

data = load_dataset("json", data_files="finred_dataset.json")

print("dataset length", len(data["train"]))
print("sample", data["train"][0])


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Find triplets in the format (subject, relation, object) in the input text.
### Input:
{data_point["text"]}
### Response:
{data_point["triplets"]}"""


CUTOFF_LEN = 2048


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


train_val = data["train"].train_test_split(test_size=20, shuffle=False)

train_data = train_val["train"].map(generate_and_tokenize_prompt)
val_data = train_val["test"].map(generate_and_tokenize_prompt)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 36
TRAIN_STEPS = 1000
MICRO_BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
OUTPUT_DIR = "experiments_finred"

model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=100,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)


def compute_metrics(pred):
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = 0
    label_str_list = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_probas = pred.predictions
    pred_ids = np.argmax(pred_probas, axis=-1)
    pred_ids[pred_ids == -100] = 0
    pred_str_list = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    pred_set, label_set = set(), set()
    with open("alpaca_log.txt", 'a') as out:
        for n, (pred, label) in enumerate(zip(label_str_list, pred_str_list)):
            if n < 50:
                out.write(f"pred: {pred}" + '\n')
                out.write(f"true: {label}" + '\n')
            pred_triplets = [triplet.strip() for triplet in pred.split(";")]
            label_triplets = [triplet.strip() for triplet in label.split(";")]
            pred_triplets = [", ".join([element.strip() for element in triplet.split()]) for triplet in pred_triplets]
            label_triplets = [", ".join([element.strip() for element in triplet.split()]) for triplet in label_triplets]
            pred_set = pred_set.union(pred_triplets)
            label_set = label_set.union(label_triplets)
        out.write("_" * 60 + '\n')

    pred_len = len(pred_set)
    true_len = len(label_set)
    right_len = len(pred_set.intersection(label_set))
    if pred_len > 0 and true_len > 0 and right_len > 0:
        precision = right_len / pred_len
        recall = right_len / true_len
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"f1": round(f1, 3)}


trainer = transformers.Trainer(
    model=model,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

model = torch.compile(model)

trainer.train()
OUTPUT_DIR = "checkpoints"
model.save_pretrained(OUTPUT_DIR)

# Loading

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, OUTPUT_DIR, torch_dtype=torch.float16)

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)
