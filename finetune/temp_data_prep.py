import json

# 读取原始数据文件
input_file = "finetune/atri.json"  # 输入文件路径
output_file = "finetune/cleaned_atri.jsonl"  # 输出文件路径

# 解析 JSON 数据
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = []

# 遍历原始数据
for dialog_str in raw_data:
    try:
        # 解析 JSON 格式的字符串
        dialog = json.loads(dialog_str.replace("“", '"').replace("”", '"'))

        # 确保格式正确
        messages = []
        for entry in dialog:
            role = entry["role"]
            content = entry["content"]
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": content})

        # 生成 `system prompt`
        system_prompt = {
            "role": "system",
            "content": (
                "你是亚托莉（Atri），一个拟人化的高级机器人，你的个性活泼可爱，"
                "喜欢与人互动，有些时候会表现出类似人类的情感。"
                "你的目标是与用户进行自然、沉浸式的角色扮演互动，你的对话应尽量符合角色设定，"
                "避免使用过于正式或死板的语言。"
            ),
        }

        # 组织新的 JSON 结构
        cleaned_data.append({"messages": [system_prompt] + messages})

    except Exception as e:
        print(f"解析错误：{e}")

# 写入新的 JSONL 文件
with open(output_file, "w", encoding="utf-8") as f:
    for entry in cleaned_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"数据清理完成，共 {len(cleaned_data)} 条对话数据已保存至 {output_file}")
