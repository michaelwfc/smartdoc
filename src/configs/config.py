"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 22:32
"""

import os
from pathlib import Path
import uuid
import torch
import logging

logger = logging.getLogger(__name__)

USE_OPENAI = True
USE_AZURE_OPENAI = False
USE_VICUNA= False
GRADIO_SERVER_PORT = 9000

# Support apps
CHATDATA = "CHATDATA"
CHATDOC = "CHATDOC"

DEFAULT_TOPIC = "DEFAULT"
SUPPORT_TOPICS = [ CHATDOC,CHATDATA,DEFAULT_TOPIC]

SUPPORT_MODELS = ["azure-openai-gpt3.5","openai-gpt3.5","vicunba-7b","llama2-7b"]
# LLM model name
LLM_MODEL = "azure_openai"
OPENAI_API_VERSION = "2023-05-15"
DEPLOYMENT_NAME = "completion-sly"

# LLM streaming reponse
STREAMING = True

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_ = LLM_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
DEVICE = f"{DEVICE_}:{DEVICE_ID}" if DEVICE_ID else DEVICE_

FLAG_USER_NAME = uuid.uuid4().hex


# Paths
DATA_DIRNAME = "data"
ASSETS_DIRNAME='assets'

DOCS_DIRNAME = "docs"

GENERATED_IMAGE_DIRNAME = "image"

WORKING_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

SRC_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(WORKING_DIR, DATA_DIRNAME)
GENERATED_IMAGE_DIR = os.path.join(DATA_DIR, GENERATED_IMAGE_DIRNAME)
KB_ROOT_PATH = DATA_DIR
TMP_FILES_DIRNAME = "tmp_files"
VS_ROOT_PATH = os.path.join(DATA_DIR,  "vector_stores")
SQLITE_ROOT_PATH = os.path.join(DATA_DIR, "sqlite_db")
UPLOAD_ROOT_PATH = os.path.join(DATA_DIR, "content")
EMBEDDING_MODEL_PATH = Path(
 r"C:\Users\\Box\Projects\mlworld\llms\text2vec-large-chinese")

TOPIC_USECASE_PATH = os.path.join(SRC_DIR,ASSETS_DIRNAME,"topic_usecase.yml")
# TOPIC_USECASE_PATH = os.path.join(DATA_DIR,"topic_data","topic_usecase.yml")


# move the nltk_data from src/assets/nltk_data to data/nltk_data
NLTK_DATA_PATH = os.path.join(DATA_DIR, "nltk_data")
TIKTOKEN_CACHE_PATH= os.path.join(DATA_DIR, "tiktoken_cache_dir","cl100k_base_tiktoken")

DOCS_PATH = os.path.join(WORKING_DIR,DOCS_DIRNAME)


# Embedding
# Embedding model name
EMBEDDING_MODEL = "text2vec"
embedding_model_dict = {
 "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
 "ernie-base": "nghuyong/ernie-3.0-base-zh",
 "text2vec-base": "shibing624/text2vec-base-chinese",
 "text2vec": EMBEDDING_MODEL_PATH,  # "GanymedeNil/text2vec-large-chinese",
}

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

# 缓存知识库数量, 如果是ChatGLM2,ChatGLM2-int4,ChatGLM2-int8模型若检索效果不好可以调成’10’
CACHED_VS_NUM = 1


# 文本分句长度
SENTENCE_SIZE = 100
# 匹配后单段上下文长度
CHUNK_SIZE = 250
# LLM input history length
LLM_HISTORY_LEN = 3
# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 5
# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0


# 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# 本地lora存放的位置
LORA_DIR = "loras/"
# LLM lora path，默认为空，如果有请直接指定文件夹路径
LLM_LORA_PATH = ""
USE_LORA = True if LLM_LORA_PATH else False

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# MOSS load in 8bit
LOAD_IN_8BIT = True

# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False


# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False

VICUNA_SERVICE_PORT = 8001
VICUNA_API_BASE =  f"http://localhost:{VICUNA_SERVICE_PORT}/v1"
VICUNA_API_KEY ="EMPTY"

# Snowflake Config
SNOWFLAKE_DATABASE_NAME = 'SNOWFLAKE_DATABASE_NAME'
SNOWFLAKE_SCHEMA_NAME = 'SNOWFLAKE_SCHEMA_NAME'
SNOWFLAKE_WAREHOUSE_NAME = 'SNOWFLAKE_WAREHOUSE_NAME'
SNOWFLAKE_ROLE_NAME = 'SNOWFLAKE_ROLE_NAME'


# 直接定义baichuan的lora完整路径即可
LORA_MODEL_PATH_BAICHUAN=""


logger.info(f"""loading model config\nllm device: {LLM_DEVICE}embedding device: {EMBEDDING_DEVICE}\n
dir: {os.path.dirname(os.path.dirname(__file__))}flagging username: {FLAG_USER_NAME}""")
