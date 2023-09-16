"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 20:20
"""
import os
import traceback
from pathlib import Path
import datetime
from typing import List, Text, Union
from tqdm import tqdm
from threading import Thread
import logging
from pypinyin import lazy_pinyin
from constants import CHATGLM_PATH, VECTOR_STORES, SOURCE_DOCUMENTS, OPENAI_EMBEDDING_MAX_INPUT_NUM, RESULT, SOURCE_DOC, \
    AADQ_API_DOC
from configs.config import KB_ROOT_PATH, SENTENCE_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K, CHUNK_SIZE, \
    EMBEDDING_MODEL, EMBEDDING_DEVICE, STREAMING, DEPLOYMENT_NAME
from textsplitters.chinese_text_splitter import ChineseTextSplitter
from utils.argument_parser import get_webui_args_parser
from utils.torch.utils import torch_gc
from utils.io import tree
from loaders.document_loader import load_langchain_docs
import langchain
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever, SVMRetriever, TFIDFRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent, AgentExecutor, AgentOutputParser
from vectorstores.vs_utils import load_vector_store, load_langchain_embeddings
# from textsplitters.chinese_text_splitter import ChineseTextSplitter
from vectorstores.faiss_vs import MyFAISS
from langchain.prompts import PromptTemplate
from models.model_loader import LoaderCheckPoint, loaderLLM
from templates.doc_agent_templates import CONTEXT_QA_TEMPLATE
from chains.doc_retrival_qa import DocRetrievalQA
from agents.base_agents import BaseAgent
from agents.openai_agents import StreamerChatOpenai, StreamerAzureChatOpenai
from utils.thread_utils import ThreadWithReturnValue
from utils.llm_utils import load_llm_model

logger = logging.getLogger(__name__)


class DocAgent:
    llm: BaseChatModel = None


    llm_model_chain: Chain = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD


    def __init__(self, embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 vs_path=None,
                 llm_model: Chain = None,
                 top_k: int = VECTOR_SEARCH_TOP_K,
                 streaming: bool = True,
                 return_source_documents: bool = True,
                 use_openai = True,
                 use_azure_openai=False,
                 use_vicuna=False,
                 use_doc_retrieval_qa=True
                 ):

        self.use_openai = use_openai
        self.use_azure_openai = use_azure_openai
        self.use_vicuna = use_vicuna

        self.streaming = streaming
        self.return_source_documents = return_source_documents

        self.use_doc_retrieval_qa = use_doc_retrieval_qa
        self.top_k = top_k
        self.llm = self._get_llm(streaming=self.streaming,
                                 use_azure_openai=self.use_azure_openai, use_vicuna=self.use_vicuna)
        self.llm_model_chain = llm_model
        self.embeddings = load_langchain_embeddings(
            use_openai=self.use_openai,
            use_azure_openai=self.use_azure_openai, use_vicuna=self.use_vicuna,
            embedding_model=embedding_model, embedding_device=embedding_device)

        self.retrieval_qa = self._build_retrieval_qa(vs_path=vs_path,
                                                     return_source_documents=self.return_source_documents,
                                                     use_doc_retrieval_qa=self.use_doc_retrieval_qa)


    def _get_llm(self, streaming=True, llm_model_name=None, callbacks=None,
                 use_openai=True,
                 use_azure_openai=False, use_vicuna=False):
        if streaming:
            # streaming: add stream_callback in ChatOpenAI
            if use_openai or use_vicuna:
                # when use vicuna
                llm = StreamerChatOpenai(temperature=0.0, streaming=streaming,
                                         callbacks=callbacks)
            elif use_azure_openai:
                # when use azure_openai
                llm = StreamerAzureChatOpenai(temperature=0.0, streaming=streaming,
                                              deployment_name=DEPLOYMENT_NAME,
                                              callbacks=callbacks)
        else:
            if use_openai or use_vicuna:
                # Create a ChatOpenAI instance for interactive chat using the OpenAI model
                llm = ChatOpenAI(temperature=0.0, streaming=streaming, callbacks=callbacks)
            elif use_azure_openai:
                llm = AzureChatOpenAI(temperature=0.0, streaming=streaming, callbacks=callbacks)

        return llm


    def _load_retriever(self, vs_path):
        """load a retriver from vs_path

       Args:
           vs_path (_type_): _description_
       """

        vector_store = load_vector_store(vs_path, self.embeddings)
        retriever = vector_store.as_retriever()
        return retriever


    def _build_retrieval_qa(self, vs_path, chain_type="stuff", return_source_documents=True,
                            callbacks=None, use_doc_retrieval_qa=False,
                            verbose=True
                            ) -> Union[RetrievalQA, DocRetrievalQA]:
        """
           chain_type= "map_reduce"/ "refine"
           #import os
           #os.environ["LANGCHAIN_TRACING_V2"] = "true"
           #os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
           #os.environ["LANGCHAIN_API_KEY"] = "..."
        :return:
        """

        llm = self.llm  # self._get_llm(streaming=self.streaming,callbacks=callbacks)
        # load a retriver
        retriever = self._load_retriever(vs_path=vs_path)
        # build the prompt template
        context_qa_template = build_context_qa_template()
        # build retraval_qa
        if use_doc_retrieval_qa == False:
            retraval_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,  # "map_reduce"/ "refine"
                retriever=retriever,
                return_source_documents=return_source_documents,
                chain_type_kwargs={"prompt": context_qa_template},
                verbose=verbose
            )
        else:
            retraval_qa = DocRetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,  # "map_reduce"/ "refine"
                retriever=retriever,
                return_source_documents=return_source_documents,
                chain_type_kwargs={"prompt": context_qa_template},
                verbose=verbose
            )
        return retraval_qa


    def run(self, query, vs_path=None, chat_history=[], model: Text = None) -> Text:
        if self.streaming:
            response = self._get_stream_answer_with_retrieval_qa(query, vs_path, chat_history, model)
        else:
            response = self.retrieval_qa.__call__({"query": query})
        return response


    def get_answer_stream_iter(self, query, vs_path, chat_history=[], streaming: bool = STREAMING,
                               use_doc_retrieval_qa=False, debug=False):


        # response = self._get_answer_with_retrieval_qa(query,vs_path)
        response = self._get_stream_answer_with_retrieval_qa(query, vs_path)
        return response


    def _get_answer_with_retrieval_qa(self, query, vs_path, chat_history=[], streaming: bool = STREAMING, debug=False):
        """get the answer by retrival qa
               Args:
                   query (_type_): _description_
                   vs_path (_type_): _description_
                   chat_history (list, optional): _description_. Defaults to [].
                   streaming (bool, optional): _description_. Defaults to STREAMING.
                   debug (bool, optional): _description_. Defaults to False.

               Returns:
                   _type_: _description_
               """

        if debug:
            langchain.debug = True
        callbacks = [StreamingStdOutCallbackHandler()]
        retrieval_qa = self._build_retrieval_qa(vs_path=vs_path, callbacks=callbacks)
        # response = retrieval_qa.__call__({"query":query})
        response = retrieval_qa.run(query)
        return response




    def _get_stream_answer_with_retrieval_qa(self, query, vs_path, chat_history=[], model: Text = None,
                                             topk_source_documents=3):
        try:
            source_doc_info = self.retrieval_qa.get_source_doc_info(inputs={"query": query})
            docs = source_doc_info[SOURCE_DOCUMENTS]

            # if AADQ_API_DOC,return result directory
            text = ""

            retrieval_qa = self.retrieval_qa
            thread = ThreadWithReturnValue(target=retrieval_qa.call_with_docs, kwargs={"inputs": {"query": query},
                                                                                       "docs": docs})
            thread.start()
            stream_callback = self.llm.stream_callback
            # print the token from stream_callback
            for token in stream_callback:
                if token is not None:
                    text += token
                    data = {
                        "text": text,
                        "error_code": 0,
                    }
            yield data

            # get the response
            response = thread.join()
            result = response.get(RESULT)
            # print the source_documents
            source_documents = response[SOURCE_DOCUMENTS]
            source_documents_text = "\n".join([f"<font color=Blue>***source doc {index}:***</font>\n{doc.page_content}"
                                               for index, doc in enumerate(source_documents[:topk_source_documents])])
            output_text = "\n\n" + source_documents_text

            for token in output_text:
                text += token
                data = {
                    "text": text,
                    "error_code": 0,
                }
            yield data
        except Exception as e:
            exception_message = traceback.format_exc()
            text = ""
            for token in exception_message:
                text += token
                data = {
                    "text": text,
                    "error_code": 999,
                }
            yield data




def doc_agent_stream_iter(agent: DocAgent, query, vs_path, streaming=True):
    """
       from fastchat.serve.api_provider import openai_api_stream_iter
       Args:
           agent (DocAgent): _description_
           query (_type_): _description_
           vs_path (_type_): _description_

       Yields:
           _type_: _description_
       """
    res = agent._get_answer_with_retrieval_qa(query=query, vs_path=vs_path)

    text = ""


    for chunk in res:
        text += chunk["result"]
        data = {
            "text": text,
            "error_code": 0,
        }
    yield data


def get_context_qa_prompt(related_docs: List[str],
                          query: str,
                          prompt_template: str = CONTEXT_QA_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace(
        "{question}", query).replace("{context}", context)


    return prompt


# Build prompt
def build_context_qa_template():
    context_qa_template = PromptTemplate.from_template(CONTEXT_QA_TEMPLATE)


    return context_qa_template


def search_result2docs(search_results):
    docs = []


    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


def init_doc_agent_from_local_model():
    """
       from init_local_doc_qa()

       Returns:
           _type_: _description_
       """
    model_path = os.getenv(CHATGLM_PATH)
    parser = get_webui_args_parser(model_path=model_path)
    args = parser.parse_args([])
    args_dict = vars(args)
    loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = loaderLLM(loaderCheckPoint=loaderCheckPoint)

    local_doc_qa = DocAgent()
    local_doc_qa.init_cfg(llm_model=llm_model_ins)


    return local_doc_qa

if __name__ == "__main__":
    streaming = True
    doc_agent = DocAgent(use_azure_openai=False, use_vicuna=True, streaming=streaming)
    query = ""
    response = doc_agent.run(query=query)
    if streaming:
        print(f"query={query}")
    for data in response:
        d = data["text"]
        print(d)
    else:
        print(f"query={query}\nresponse={response}")