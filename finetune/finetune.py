import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# 1. 数据处理
def load_and_process_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_data = []
    for dialog_str in raw_data:
        # 转换中文引号为英文引号
        dialog_str = dialog_str.replace("“", '"').replace("”", '"')
        try:
            dialog = json.loads(dialog_str)
            messages = []
            for msg in dialog:
                # 确保角色字段符合要求
                role = (
                    msg["role"]
                    .replace("assistant", "assistant")
                    .replace("user", "user")
                )
                messages.append({"role": role, "content": msg["content"].strip()})
            processed_data.append({"messages": messages})
        except json.JSONDecodeError:
            print(f"解析失败: {dialog_str}")

    return Dataset.from_list(processed_data)


# 加载数据集
dataset = load_and_process_data("finetune/atri.json")

# 2. 加载模型和分词器
model_name = "Qwen/Qwen1.5-7B-Chat"  # 可替换为其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", trust_remote_code=True
)


# 3. 数据预处理函数
def format_conversation(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


tokenized_dataset = dataset.map(
    format_conversation, remove_columns=dataset.column_names
)

# 4. 配置 LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 训练参数设置
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="adamw_torch",
)

# 6. 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 8. 开始训练
trainer.train()

# 9. 保存模型
model.save_pretrained("./atri_lora")
tokenizer.save_pretrained("./atri_lora")
