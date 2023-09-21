"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 20:35
"""
from typing import List, Dict, Text, Optional, Any, Iterator
from fastchat.serve.api_provider import openai_api_stream_iter
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models.openai import ChatOpenAI, convert_openai_messages
from langchain.chat_models import AzureChatOpenAI
from agents.base_agents import BaseAgent
from constants import GPT_3DOT5_TURBO
from callbacks.stream_callbacks import StreamingOutCallbackHandler
from utils.llm_utils import load_llm_model


class OpenaiAgent(BaseAgent):
    def __init__(self):
        self.streaming = True
        self.use_openai_api = False
        self.use_chat_openai_api = True

    def run(self, prompt: List[Dict], temperature: float, top_p: int, max_new_tokens: int,
            model_name: Text = GPT_3DOT5_TURBO):
        if self.use_openai_api:
            output = self._run_openai_api(prompt, temperature, top_p, max_new_tokens, model_name)
        elif self.use_chat_openai_api:
            output = self._run_langchain_chat_openai_api(prompt, temperature, top_p, max_new_tokens, model_name)
        else:
            raise ValueError(f"mode not support.")
        return output


    def _run_openai_api(self, prompt: List[Dict], temperature: float, top_p: int, max_new_tokens: int,
                        model_name: Text = GPT_3DOT5_TURBO):
        output = openai_api_stream_iter(model_name=model_name, messages=prompt, temperature=temperature, \
                                        top_p=top_p, max_new_tokens=max_new_tokens)


        return output


    def _run_langchain_chat_openai_api(self, prompt: List[Dict], temperature: float, top_p: int, max_new_tokens: int,
                                       model_name: Text = GPT_3DOT5_TURBO, use_azure_openai=True):
        """use ChatOpenAI.stream() method to get the iterator of chunk

               Args:
                   prompt (List[Dict]): _description_
                   temperature (float): _description_
                   top_p (int): _description_
                   max_new_tokens (int): _description_
                   model_name (Text, optional): _description_. Defaults to GPT_3DOT5_TURBO.

               Yields:
                   _type_: _description_
               """
        if use_azure_openai:
            chat_openai = load_llm_model(use_azure_openai=True)
            # res = chat_openai.stream(input=prompt)
        else:
            chat_openai = ChatOpenAI(model=model_name, temperature=temperature, streaming=self.streaming)
            messages = convert_openai_messages(messages=prompt)
            res = chat_openai.stream(input=messages)
            text = ""
            for chunk in res:
                text += chunk.content
                data = {
                    "text": text,
                    "error_code": 0,
                }
                yield data


class StreamerChatOpenai(ChatOpenAI):
    """add stream_callback to record the token

       Args:
           ChatOpenAI (_type_): _description_
       """
    stream_callback: StreamingOutCallbackHandler = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use StreamingOutCallbackHandlerV2
        self.stream_callback = StreamingOutCallbackHandler()
        # add StreamingStdOutCallbackHandler to callbacks
        self.callbacks = [self.stream_callback, StreamingStdOutCallbackHandler()]


class StreamerAzureChatOpenai(AzureChatOpenAI, StreamerChatOpenai):
    pass