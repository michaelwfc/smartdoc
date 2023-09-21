"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 20:32
"""

import openai
from typing import Text, List, Dict
from constants import  GPT_3DOT5_TURBO,GPT_35_TURBO_16K
from configs.config import DEPLOYMENT_NAME
from utils.environments import set_gpt_env


def get_response_from_openai_api(prompt: Text, temperature=0.5, model_engine=GPT_3DOT5_TURBO) -> Text:
    """
    prompt: which is a variable representing the text input to the function.
    temperature: how deterministic the output of the model is. A high temperature gives the model more freedom to sample outputs.
    top_p: the distribution to select the outputs from.
    max_tokens: the limit for the number of words to be returned.
    frequency_penalty and presence_penalty: both are parameters which penalise the model for returning outputs which appear often.
    """
    # Generate a response
    completion = openai.Completion.create(
     engine=model_engine,
     prompt=prompt,
     max_tokens=1024,
     n=1,
     stop=None,
     temperature=temperature,
        )
    response = completion.choices[0].text

    return response


def get_completion(prompt: Text, model= GPT_3DOT5_TURBO):
    """
    in OpenAI Chat Completion API, a chat message can be associated with the AI, human or system role.
    The model is supposed to follow instruction from system chat message more closely.
    messages = [
    {"role": "system", "content": "You are an assistant..."},
    {"role": "user", "content": "Tell me a joke"},
    {"role": "assistant", "content": "Why did the chicken..."},
    ]

    sets behavior of assistant
    assistant: chat model
    user : you
    :param prompt:
    :param model:
    :return:
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_completion_from_messages(messages: List[Dict], model= GPT_3DOT5_TURBO, temperature=0, use_azure_openai=True):
    """

    :param messages:
        messages =  [
        {'role':'system',
         'content': CLASSIFY_SYSTEM_MESSAGE},
        {'role':'user',
         'content': f"{DELIMITER}{user_message}{DELIMITER}"},
    ]
    :param model:
    :param temperature:
    :return:
    """
    if use_azure_openai:
        deployment_name = DEPLOYMENT_NAME if use_azure_openai else None
        response = openai.ChatCompletion.create(
        model=model,
        engine=deployment_name,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
        )
    else:
        response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
        )
        #     print(str(response.choices[0].message))
        content = response.choices[0].message["content"]

        token_dict = {
        'prompt_tokens': response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens'],
        'total_tokens': response['usage']['total_tokens'],
        }
    return content


def get_response_from_moderation_api(prompt: Text, model="gpt-3.5-turbo"):
    response = openai.Moderation.create(
    input=prompt, model=model
    )
    moderation_output = response["results"][0]
    return moderation_output




if __name__ == '__main__':
     from vectorstores.vs_utils import load_langchain_embeddings
     from configs.config import USE_OPENAI
     set_gpt_env(use_openai=USE_OPENAI)
     messages = [
            {'role': 'system',
     'content': """You are an assistant who  responds in the style of Dr Seuss."""},
            {'role': 'user',
     'content': """write me a very short poem about a happy carrot"""},
        ]
     response = get_completion_from_messages(messages,temperature=1,use_azure_openai=False)
     print(response)

     # query="write me a very short poem about a happy carrot"
     # embeddings = load_langchain_embeddings(use_azure_openai=use_azure_openai,use_vicuna=use_vicuna)
     # embeddding = embeddings.embed_query(query)
     # print(embeddings)
     # print(f"query={query},embedding={embeddding} ")