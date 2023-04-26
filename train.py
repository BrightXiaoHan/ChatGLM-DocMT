import transformers
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from transformers import AutoModel, AutoTokenizer

from data import DocMTDataset, load, collate_fn

MODEL_NAME = "THUDM/chatglm-6b"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_key_value"],
    bias="none",
)

model = AutoModel.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282


train_pairs = load("data/test.jsonl", tokenizer)
train_data = DocMTDataset(train_pairs)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        output_dir="outputs",
    ),
    data_collator=collate_fn,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
