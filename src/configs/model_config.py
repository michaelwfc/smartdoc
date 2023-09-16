"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: model_config.py
@description: 
@time: 2023/9/16 9:43
"""
import os

#supported LLM models
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
# 在以下字典中修改属性值，以指定本地 LLM 模型存储位置
# 如将 "chatglm-6b" 的 "local_model_path" 由 None 修改为 "User/Downloads/chatglm-6b"
# 此处请写绝对路径,且路径中必须包含repo-id的模型名称，因为FastChat是以模型名匹配的
LLM_MODEL_DICT = {
    "chatglm-6b-int4-qe": {
        "name": "chatglm-6b-int4-qe",
        "pretrained_model_name": "THUDM/chatglm-6b-int4-qe",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm-6b-int4": {
        "name": "chatglm-6b-int4",
        "pretrained_model_name": "THUDM/chatglm-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm-6b-int8": {
        "name": "chatglm-6b-int8",
        "pretrained_model_name": "THUDM/chatglm-6b-int8",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    # langchain-ChatGLM 用户“帛凡” @BoFan-tunning 基于ChatGLM-6B 训练并提供的权重合并模型和 lora 权重文件 chatglm-fitness-RLHF
    # 详细信息见 HuggingFace 模型介绍页 https://huggingface.co/fb700/chatglm-fitness-RLHF
    # 使用该模型或者lora权重文件，对比chatglm-6b、chatglm2-6b、百川7b，甚至其它未经过微调的更高参数的模型，在本项目中，总结能力可获得显著提升。
    "chatglm-fitness-RLHF": {
        "name": "chatglm-fitness-RLHF",
        "pretrained_model_name": "fb700/chatglm-fitness-RLHF",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm2-6b": {
        "name": "chatglm2-6b",
        "pretrained_model_name": "THUDM/chatglm2-6b",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm2-6b-32k": {
        "name": "chatglm2-6b-32k",
        "pretrained_model_name": "THUDM/chatglm2-6b-32k",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    # 注：chatglm2-cpp已在mac上测试通过，其他系统暂不支持
    "chatglm2-cpp": {
        "name": "chatglm2-cpp",
        "pretrained_model_name": "cylee0909/chatglm2cpp",
        "local_model_path": None,
        "provides": "ChatGLMCppLLMChain"
    },
    "chatglm2-6b-int4": {
        "name": "chatglm2-6b-int4",
        "pretrained_model_name": "THUDM/chatglm2-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm2-6b-int8": {
        "name": "chatglm2-6b-int8",
        "pretrained_model_name": "THUDM/chatglm2-6b-int8",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatyuan": {
        "name": "chatyuan",
        "pretrained_model_name": "ClueAI/ChatYuan-large-v2",
        "local_model_path": None,
        "provides": "MOSSLLMChain"
    },
    "moss": {
        "name": "moss",
        "pretrained_model_name": "fnlp/moss-moon-003-sft",
        "local_model_path": None,
        "provides": "MOSSLLMChain"
    },
    "moss-int4": {
        "name": "moss",
        "pretrained_model_name": "fnlp/moss-moon-003-sft-int4",
        "local_model_path": None,
        "provides": "MOSSLLM"
    },
    "vicuna-13b-hf": {
        "name": "vicuna-13b-hf",
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "LLamaLLMChain"
    },
    "vicuna-7b-hf": {
        "name": "vicuna-13b-hf",
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "LLamaLLMChain"
    },
    # 直接调用返回requests.exceptions.ConnectionError错误，需要通过huggingface_hub包里的snapshot_download函数
    # 下载模型，如果snapshot_download还是返回网络错误，多试几次，一般是可以的，
    # 如果仍然不行，则应该是网络加了防火墙(在服务器上这种情况比较常见)，基本只能从别的设备上下载，
    # 然后转移到目标设备了.
    "bloomz-7b1": {
        "name": "bloomz-7b1",
        "pretrained_model_name": "bigscience/bloomz-7b1",
        "local_model_path": None,
        "provides": "MOSSLLMChain"

    },
    # 实测加载bigscience/bloom-3b需要170秒左右，暂不清楚为什么这么慢
    # 应与它要加载专有token有关
    "bloom-3b": {
        "name": "bloom-3b",
        "pretrained_model_name": "bigscience/bloom-3b",
        "local_model_path": None,
        "provides": "MOSSLLMChain"

    },
    "baichuan-7b": {
        "name": "baichuan-7b",
        "pretrained_model_name": "baichuan-inc/baichuan-7B",
        "local_model_path": None,
        "provides": "MOSSLLMChain"
    },
    "Baichuan-13b-Chat": {
        "name": "Baichuan-13b-Chat",
        "pretrained_model_name": "baichuan-inc/Baichuan-13b-Chat",
        "local_model_path": None,
        "provides": "BaichuanLLMChain"
    },
    # llama-cpp模型的兼容性问题参考https://github.com/abetlen/llama-cpp-python/issues/204
    "ggml-vicuna-13b-1.1-q5": {
        "name": "ggml-vicuna-13b-1.1-q5",
        "pretrained_model_name": "lmsys/vicuna-13b-delta-v1.1",
        # 这里需要下载好模型的路径,如果下载模型是默认路径则它会下载到用户工作区的
        # /.cache/huggingface/hub/models--vicuna--ggml-vicuna-13b-1.1/
        # 还有就是由于本项目加载模型的方式设置的比较严格，下载完成后仍需手动修改模型的文件名
        # 将其设置为与Huggface Hub一致的文件名
        # 此外不同时期的ggml格式并不兼容，因此不同时期的ggml需要安装不同的llama-cpp-python库，且实测pip install 不好使
        # 需要手动从https://github.com/abetlen/llama-cpp-python/releases/tag/下载对应的wheel安装
        # 实测v0.1.63与本模型的vicuna/ggml-vicuna-13b-1.1/ggml-vic13b-q5_1.bin可以兼容
        "local_model_path": f'''{"/".join(os.path.abspath(__file__).split("/")[:3])}/.cache/huggingface/hub/models--vicuna--ggml-vicuna-13b-1.1/blobs/''',
        "provides": "LLamaLLMChain"
    },

    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-chatglm-6b": {
        "name": "chatglm-6b",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm-6b",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8000/v1",  # "name"修改为fastchat服务中的"api_base_url"
        "api_key": "EMPTY"
    },
        # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-chatglm-6b-int4": {
        "name": "chatglm-6b-int4",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm-6b-int4",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8001/v1",  # "name"修改为fastchat服务中的"api_base_url"
        "api_key": "EMPTY"
    },
    "fastchat-chatglm2-6b": {
        "name": "chatglm2-6b",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm2-6b",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8000/v1"  # "name"修改为fastchat服务中的"api_base_url"
    },

    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-vicuna-13b-hf": {
        "name": "vicuna-13b-hf",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8000/v1",  # "name"修改为fastchat服务中的"api_base_url"
        "api_key": "EMPTY"
    },
    # 调用chatgpt时如果报出： urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.openai.com', port=443):
    #  Max retries exceeded with url: /v1/chat/completions
    # 则需要将urllib3版本修改为1.25.11
    # 如果依然报urllib3.exceptions.MaxRetryError: HTTPSConnectionPool，则将https改为http
    # 参考https://zhuanlan.zhihu.com/p/350015032

    # 如果报出：raise NewConnectionError(
    # urllib3.exceptions.NewConnectionError: <urllib3.connection.HTTPSConnection object at 0x000001FE4BDB85E0>:
    # Failed to establish a new connection: [WinError 10060]
    # 则是因为内地和香港的IP都被OPENAI封了，需要切换为日本、新加坡等地
    "openai-chatgpt-3.5": {
        "name": "gpt-3.5-turbo",
        "pretrained_model_name": "gpt-3.5-turbo",
        "provides": "FastChatOpenAILLMChain",
        "local_model_path": None,
        "api_base_url": "https://api.openai.com/v1",
        "api_key": ""
    },

}