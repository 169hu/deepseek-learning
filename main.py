import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的变量
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "请用一句话介绍你自己。"}
    ],
    stream=False
)

print(response.choices[0].message.content)