import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from new_agent.app.args import zhipu_llm

if __name__ == "__main__":
    # 测试智谱大模型
    response = zhipu_llm.invoke("你好")
    print(response)