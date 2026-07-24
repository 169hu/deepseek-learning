from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# ===== 1. 准备 Few-shot 示例（这就是“教具”） =====
examples = [
    {"input": "小明，数学95，语文88", "output": {"name": "小明", "scores": {"数学": 95, "语文": 88}}},
    {"input": "小红，英语92，科学85", "output": {"name": "小红", "scores": {"英语": 92, "科学": 85}}}
]

# ===== 2. 构建“带示例”的 System Prompt =====
system_prompt = "你是一个数据提取助手。请严格按照以下示例的 JSON 格式输出，不要添加任何额外文字。\n"
for ex in examples:
    system_prompt += f"输入：{ex['input']} → 输出：{ex['output']}\n"

# ===== 3. 用户新输入（从未见过的新数据） =====
new_input = "小刚，数学78，英语82，历史90"

# ===== 4. 调用模型 =====
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": new_input}
    ],
    temperature=0.1  # 越低越稳定
)

print("提取结果：")
print(response.choices[0].message.content)