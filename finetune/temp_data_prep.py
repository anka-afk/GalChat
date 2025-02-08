import json

input_file = "finetune/atri.json"
output_file = "finetune/cleaned_atri.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = []
for dialog_str in raw_data:
    try:
        dialog = json.loads(dialog_str.replace("“", '"').replace("”", '"'))
        messages = []
        for entry in dialog:
            role = entry["role"]
            content = entry["content"]
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": content})

        system_prompt = {
            "role": "system",
            "content": (
                "你是亚托莉（Atri），一个拟人化的高级机器人，你的个性活泼可爱，"
                "喜欢与人互动，有些时候会表现出类似人类的情感。"
                "你的目标是与用户进行自然、沉浸式的角色扮演互动，你的对话应尽量符合角色设定，"
                "避免使用过于正式或死板的语言。"
            ),
        }

        cleaned_data.append({"messages": [system_prompt] + messages})

    except Exception as e:
        print(f"解析错误：{e}")

with open(output_file, "w", encoding="utf-8") as f:
    for entry in cleaned_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"数据清理完成，共 {len(cleaned_data)} 条对话数据已保存至 {output_file}")
