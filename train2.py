import os

import torch
import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import data

checkpoint = "THUDM/chatglm-6b"

model_id = "finetune_test"

mixed_precision = "bf16"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_key_value"],
    bias="none",
)

LR = 1e-4
BATCH = 1
MAX_LENGTH = 2048
NUM_EPOCHS = 3
accumulate_step = 8
warm_up_ratio = 0.1


deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(
    mixed_precision=mixed_precision,
    deepspeed_plugin=deepspeed_plugin,
    log_with="tensorboard",
    project_dir="outputs/",
)
device = accelerator.device
data.device = device


with accelerator.main_process_first():

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, revision="main"
    )
    model = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=True, revision="main"
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)


accelerator.wait_for_everyone()

model.use_cache = False
model.gradient_checkpointing = False


accelerator.print("Start to process data")


with accelerator.main_process_first():
    pairs = data.load("./data/test.jsonl", tokenizer)
train_dataset = data.DocMTDataset(pairs)
train_dataloader = DataLoader(
    dataset=train_dataset,
    collate_fn=data.collate_fn,
    shuffle=True,
    batch_size=BATCH,
)

accelerator.wait_for_everyone()


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(len(train_dataloader) // accumulate_step * NUM_EPOCHS),
)
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)


accelerator.init_trackers(model_id, {})

total_effective_step = 0

for epoch in range(NUM_EPOCHS):

    batch_loss = 0
    effective_step = 0

    for step, batch in enumerate(t := tqdm.tqdm(train_dataloader)):

        outputs = model(**batch)

        loss_d = outputs.loss.detach().cpu().float().item()
        batch_loss += loss_d

        loss = outputs.loss / accumulate_step
        accelerator.backward(loss)

        if (step + 1) % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            effective_step += 1

            gathered_batch_loss = accelerator.gather(
                (torch.tensor(batch_loss, device=device))
            )

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "train_loss": gathered_batch_loss.mean().item()
                        / accumulate_step,
                        "epoch": epoch,
                    },
                    step=total_effective_step + effective_step,
                )

            t.set_description(
                f"loss: {gathered_batch_loss.mean().item() / accumulate_step}"
            )
            batch_loss = 0

    accelerator.wait_for_everyone()

    total_effective_step += effective_step

    if accelerator.is_main_process:
        os.makedirs(f"saved/{model_id}", exist_ok=True)
        accelerator.save(
            accelerator.unwrap_model(model),
            f"saved/{model_id}/{model_id}_epoch_{epoch}.pt",
        )

    accelerator.wait_for_everyone()
