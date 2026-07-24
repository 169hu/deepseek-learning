import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# ===== 1. 加载环境变量 =====
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ===== 2. 定义一个“本地工具函数”（模拟真实天气查询） =====
def get_current_weather(location):
    """根据地点返回模拟天气（实际项目中可替换为真实API请求）"""
    # 模拟一个天气数据库
    weather_db = {
        "上海": "28°C，多云转阴，东南风3级",
        "北京": "32°C，晴，空气质量良",
        "深圳": "30°C，雷阵雨，注意带伞",
        "成都": "26°C，阴天，湿度较大"
    }
    # 如果查不到，返回默认值
    return weather_db.get(location, f"{location}的天气：20°C，晴（模拟数据）")

# ===== 3. 定义工具的“描述蓝图”（告诉模型这个工具能干什么） =====
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "查询指定地点的当前天气情况",  # 模型根据这个描述决定是否调用
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：上海、北京"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# ===== 4. 核心 Agent 流程 =====
def run_agent(user_query):
    print(f"👤 用户问：{user_query}")

    # 第一步：把用户问题发给模型，并带上“工具说明书”
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": user_query}],
        tools=tools,  # 关键参数：挂载工具
        tool_choice="auto"  # 让模型自己决定要不要用工具
    )

    # 获取模型返回的消息
    message = response.choices[0].message

    # 第二步：判断模型是否想要调用工具
    if message.tool_calls:
        # 模型说：“我要调用工具！”
        tool_call = message.tool_calls[0]  # 取第一个工具调用
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)  # 解析参数 JSON

        print(f"🧠 模型决定调用工具：{function_name}")
        print(f"📦 提取的参数：{arguments}")

        # 第三步：执行本地函数（拿到真实数据）
        if function_name == "get_current_weather":
            result = get_current_weather(arguments["location"])
            print(f"🌤️ 工具返回结果：{result}")

            # 第四步：把工具返回的结果包装成消息，再发给模型
            # 注意：角色是 "tool"，并且要带上 tool_call_id
            second_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": user_query},
                    message,  # 模型第一次的回复（包含 tool_calls）
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                ]
            )
            # 模型根据工具结果生成最终回答
            final_answer = second_response.choices[0].message.content
            print(f"🤖 最终回答：{final_answer}")
    else:
        # 如果模型觉得不需要工具，直接输出回答
        print(f"🤖 直接回答：{message.content}")

# ===== 5. 启动测试 =====
if __name__ == "__main__":
    print("=" * 40)
    # 测试1：需要调用工具的问题
    run_agent("上海今天的天气怎么样？")
    print("\n" + "=" * 40)
    # 测试2：不需要调用工具的问题
    run_agent("你好，你是谁？")
    print("=" * 40)