from langchain.callbacks.manager import AsyncCallbackManager

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from langchain_community.chat_models import ChatZhipuAI
from new_agent.callbacks.base import AgentExecutorAsyncIteratorCallbackHandler
from new_agent.agents.all_agents import init_rag_agent, init_search_agent
from new_agent.utils.memory import MemoryChain
from typing import Dict, Any
from fastapi import FastAPI

import json
from local import Agent_DB
import uvicorn




class ChatAPP:
    def __init__(self):
        pass