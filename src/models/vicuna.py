"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: vicuna.py
@time: 2023/8/30 22:40
"""
from typing import Any, Dict, Optional, Union, Tuple, Literal, Set, Sequence
import torch
from pydantic import Field
from constants import VICUNA_PATH
from utils.environments import load_env_variables_from_dotenv
from apis.openai_apis import get_completion_from_messages
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from configs.config import VICUNA_SERVICE_PORT,VICUNA_API_BASE,VICUNA_API_KEY
from constants import GPT_3DOT5_TURBO



def run_restful_api_server_as_openai_api(port=VICUNA_SERVICE_PORT,model="D:\\llms\\vicuna-7b"):
    """
    https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md
    1. launch the controller
    python -m fastchat.serve.controller &
    2. launch the model worker(s)
    python -m fastchat.serve.model_worker --model-path D:\\llms\\vicuna-7b --model-names vicuna &
    3. launch the RESTful API server
    python -m fastchat.serve.openai_api_server --host localhost --port 8001 &
    :return:
    """
    # from fastchat.serve.controller import Controller
    # from fastchat.serve.model_worker import ModelWorker
    # from fastchat.serve.openai_api_server import AppSettings

    openai.api_key =  VICUNA_API_KEY
    openai.api_base = VICUNA_API_BASE
    response = get_completion_from_messages(messages='hello',model= model)
    print(f"response={response}")



def load_vicuna_model_from_hf_api(temperature=0.7, repetition_penalty=1.0,max_new_tokens=512):
    """
    https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/huggingface_api.py
    :return:
    """
    from fastchat.model import load_model, get_conversation_template
    env = load_env_variables_from_dotenv()
    model_path =  env[VICUNA_PATH]
    # supported LLM models
    model,tokenizer = load_model(model_path=model_path,device='cuda',num_gpus=1)
    # tokenizer = AutoTokenizer.from_pretrained(vicuna_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(vicuna_path, trust_remote_code=True).half().cuda()
    model.eval()

    msg = "hello"
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
    do_sample=True,
    temperature=temperature,
    repetition_penalty=repetition_penalty,
    max_new_tokens=max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")

    return model,tokenizer


class ChatVicuna(ChatOpenAI):
    client: Any  #: :meta private:
    model_name: str = Field(default=GPT_3DOT5_TURBO, alias="model")
    openai_api_key: Optional[str] =  "EMPTY"
    """Base URL path for API requests, 
    leave blank if not using a proxy or service emulator."""
    openai_api_base: Optional[str] = VICUNA_API_BASE

    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    openai_organization: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    tiktoken_model_name: Optional[str] = None
    """The model name to pass to tiktoken when using this class. 
    Tiktoken is used to count the number of tokens in documents to constrain 
    them to be under a certain limit. By default, when set to None, this will 
    be the same as the embedding model name. However, there are some cases 
    where you may want to use this Embedding class with a model name not 
    supported by tiktoken. This can include when using Azure embeddings or 
    when using one of the many model providers that expose an OpenAI-like 
    API but with different models. In those cases, in order to avoid erroring 
    when tiktoken is called, you can specify a model name to use here."""


class VicunaEmbeddings(OpenAIEmbeddings):
    client: Any  #: :meta private:
    model: str  = GPT_3DOT5_TURBO
    deployment: str = model  # to support Azure OpenAI Service custom deployment names
    openai_api_version: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = VICUNA_API_BASE
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout in seconds for the OpenAPI request."""
    headers: Any = None
    tiktoken_model_name: Optional[str] = GPT_3DOT5_TURBO