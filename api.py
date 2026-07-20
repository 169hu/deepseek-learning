from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# ==================== 1. 加载环境变量 ====================
load_dotenv()

# ==================== 2. 配置日志（在 load_dotenv() 之后） ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 3. 初始化 OpenAI 客户端 ====================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ==================== 4. 创建 FastAPI 应用 ====================
app = FastAPI(title="DeepSeek 工具箱 API", version="1.0")

# ==================== 5. 定义请求体格式 ====================
class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "英文"

# ==================== 6. 根路径测试 ====================
@app.get("/")
def root():
    logger.info("根路径被访问")
    return {"message": "DeepSeek API 服务已启动！"}

# ==================== 7. 翻译接口（带日志） ====================
@app.post("/translate/")
def translate(request: TranslateRequest):
    start_time = time.time()
    logger.info(f"收到翻译请求: text='{request.text}', target='{request.target_lang}'")

    try:
        lang_map = {
            "英文": "English", "中文": "Chinese", "日文": "Japanese",
            "韩文": "Korean", "法文": "French", "德文": "German"
        }
        target = lang_map.get(request.target_lang, "English")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"你是莎士比亚时代的英国诗人。请将中文翻译成极具古典文学色彩的{target}，不准意译，必须直译。"},
                {"role": "user", "content": request.text}
            ]
        )
        result = response.choices[0].message.content
        elapsed = time.time() - start_time
        logger.info(f"翻译成功，耗时 {elapsed:.2f}s，结果长度 {len(result)} 字符")
        return {"code": 200, "result": result, "elapsed": f"{elapsed:.2f}s"}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"翻译失败，耗时 {elapsed:.2f}s，错误: {str(e)}")
        return {"code": 500, "error": str(e)}