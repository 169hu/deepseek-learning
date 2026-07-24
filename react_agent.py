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

# ===== 2. 定义工具函数 =====

# 工具1：查询天气
def get_current_weather(location):
    weather_db = {
        "上海": "28°C，多云转阴，东南风3级",
        "北京": "32°C，晴，空气质量良",
        "深圳": "30°C，雷阵雨，注意带伞",
        "成都": "26°C，阴天，湿度较大",
        "广州": "31°C，多云，微风"
    }
    return weather_db.get(location, f"{location}：20°C，晴（模拟数据）")

# 工具2：查询汇率（新增）
def get_exchange_rate(from_currency, to_currency):
    """模拟汇率查询"""
    # 模拟汇率数据（实际项目中可调用真实 API）
    rates = {
        ("人民币", "美元"): 0.14,
        ("美元", "人民币"): 7.15,
        ("人民币", "日元"): 20.50,
        ("日元", "人民币"): 0.049,
        ("人民币", "欧元"): 0.13,
        ("欧元", "人民币"): 7.70,
    }
    return rates.get((from_currency, to_currency), 0.00)


# ===== 3. 定义工具的“使用说明书”（多工具） =====
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "查询指定地点的当前天气情况",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "查询两种货币之间的汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "description": "源货币代码，例如：人民币、美元、日元、欧元"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "目标货币代码，例如：人民币、美元、日元、欧元"
                    }
                },
                "required": ["from_currency", "to_currency"]
            }
        }
    }
]


# ===== 4. 核心 ReAct Agent =====

def run_react_agent(user_query, max_steps=5):
    """
    支持多步推理的 ReAct Agent
    """
    print(f"👤 用户：{user_query}")
    print("=" * 50)

    # 历史消息（包含所有推理和工具调用记录）
    messages = [{"role": "user", "content": user_query}]
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n🔄 第 {step} 步推理...")

        # 调用模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)  # 保存模型的消息到历史

        # ===== 情况1：模型决定调用工具 =====
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"   🧠 调用工具：{function_name}")
                print(f"   📦 参数：{arguments}")

                # 执行对应的本地函数
                if function_name == "get_current_weather":
                    result = get_current_weather(arguments["location"])
                elif function_name == "get_exchange_rate":
                    result = get_exchange_rate(
                        arguments["from_currency"],
                        arguments["to_currency"]
                    )
                    result = f"1 {arguments['from_currency']} = {result} {arguments['to_currency']}"
                else:
                    result = "未知工具"

                print(f"   📊 工具返回：{result}")

                # 把工具结果添加到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # ===== 情况2：模型直接回答（不再调用工具） =====
        else:
            print("\n✅ 任务完成！")
            print("=" * 50)
            print(f"🤖 最终回答：{message.content}")
            return message.content

    # 如果超出最大步数，强制结束
    print("\n⚠️ 达到最大推理步数，强制结束。")
    return "抱歉，任务过于复杂，无法在限定步数内完成。"


# ===== 5. 测试 =====
if __name__ == "__main__":
    print("=" * 50)
    print("🧠 ReAct Agent 测试（多工具 + 多步推理）")
    print("=" * 50)

    # 测试1：单步单工具（天气查询）
    run_react_agent("上海今天天气怎么样？")

    print("\n" + "=" * 50)

    # 测试2：多步多工具（需要先查天气，再对比？这里设计一个需要两步的问题）
    run_react_agent("100元人民币能换多少美元？")

    print("\n" + "=" * 50)

    # 测试3：多步工具（先查天气，再对比——让模型自己决定步骤）
    run_react_agent("北京和上海哪个更热？")