"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: llm_utils.py
@time: 2023/8/30 22:49
"""
from typing import Union
from configs.config import DEPLOYMENT_NAME
from langchain.llms import OpenAI,BaseLLM
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from models.vicuna import ChatVicuna


def load_llm_model(use_vicuna=False, use_openai=False, use_azure_openai=True, temperature=0.0)->Union[BaseLLM, ChatOpenAI]:
    if use_vicuna:
        return ChatVicuna(temperature=temperature)
    if use_azure_openai:
        return AzureChatOpenAI(temperature=temperature, deployment_name=DEPLOYMENT_NAME)
    if use_openai:
        return ChatOpenAI(temperature= temperature)
