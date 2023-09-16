"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: vs_builders.py
@time: 2023/8/30 23:02
"""
import os
from pathlib import Path
from typing import List, Text, Dict

from langchain.vectorstores import VectorStore
from tqdm import tqdm
from pypinyin import lazy_pinyin
import datetime
from langchain.docstore.document import Document
from constants import SOURCE_DOC
from utils.io import tree
from utils.torch.utils import torch_gc
from loaders.document_loader import load_langchain_docs, add_source_to_metadata
from vectorstores.vs_utils import load_vector_store, load_langchain_embeddings
from constants import VECTOR_STORES
from configs.config import KB_ROOT_PATH, SENTENCE_SIZE
from vectorstores.faiss_vs import MyFAISS

import logging

logger = logging.getLogger(__name__)


class VectorStoreBuilder():

    def __init__(self, use_openai=True, use_azure_openai: bool = False, use_vicuna=False,
                 embedding_model: Text = None, embedding_device=None) -> None:
        self.embeddings = load_langchain_embeddings(
            use_openai = use_openai,
            use_azure_openai=use_azure_openai, use_vicuna=use_vicuna,
            embedding_model=embedding_model, embedding_device=embedding_device)

    def init_knowledge_vector_store(self, filepath: str or List[str], source: Text = None,
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE,mode="not_add"):
        try:
            docs, loaded_files = self._load_docs(filepath=filepath, sentence_size=sentence_size)
            if source:
                docs = [add_source_to_metadata(doc, source) for doc in docs]
            # build the vs
            if len(docs) > 0:
                logger.info("start to build vectorstore")
                if not vs_path:
                    if isinstance(filepath, list):
                        file_path = filepath[0]
                    else:
                        file_path = filepath

                    file_stem = Path(file_path).stem
                    vs_path = os.path.join(KB_ROOT_PATH, VECTOR_STORES,
                                           f"""{"".join(lazy_pinyin(file_stem))}_FAISS""",
                                           )
                    # f"""{"".join(lazy_pinyin(file_stem))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""",

                if vs_path and os.path.isdir(vs_path) and "index.faiss" in os.listdir(vs_path):
                    vector_store = load_vector_store(vs_path, self.embeddings)
                    if mode == "add":
                        vector_store.add_documents(docs)
                    torch_gc()
                else:
                    # ERROR: OpenAI API error received: 'Too many inputs. The max number of inputs is 16.
                    # docs = docs[:OPENAI_EMBEDDING_MAX_INPUT_NUM]
                    # vector_store = MyFAISS.from_documents(docs, self.embeddings)  # docs 为Document列表
                    vector_store = MyFAISS.from_documents_with_batch(docs, self.embeddings)
                    torch_gc()

                vector_store.save_local(vs_path)
                return vs_path, loaded_files
            else:
                logger.error("filed uploaded failed, please check.")
                raise ValueError("filed uploaded failed, please check.")
        except Exception as e:
            raise e


    def _load_docs(self, filepath: str or List[str], sentence_size=SENTENCE_SIZE):
        loaded_files = []
        failed_files = []
        # load file(s) to docs
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                logger.error(f"filepath={filepath} not exists.")
                # return None
                raise FileExistsError(f"filepath={filepath} not exists.")
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_langchain_docs(filepath, sentence_size)
                    logger.info(f"{file} success loaded.")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e, exc_info=True)
                    logger.info(f"{file} fail to load.")
                    raise e
            elif os.path.isdir(filepath):
                docs = []
                for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc="loading file directory"):
                    try:
                        docs += load_langchain_docs(fullfilepath,
                                                    sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_langchain_docs(file)
                    logger.info(f"{file} load succfully.")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} fail to load")
                    failed_files.append(file)

        if len(failed_files) > 0:
            logger.info("the following files are faild to load：")
            for file in failed_files:
                logger.info(f"{file}\n")

        logger.info(f"finished load files to docs.")
        return docs, loaded_files


    def add_doc_to_knowledge(self, vs_path, filepath: str or List[str], source: Text, sentence_size=set):
        try:
            if not vs_path or not filepath or not source:
                logger.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [source]
            docs = load_langchain_docs(filepath=filepath, sentence_size=sentence_size)

            # add source to metadata
            if os.path.isdir(vs_path) and os.path.isfile(vs_path + "/index.faiss"):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                return None, [source]
            torch_gc()
            vector_store.save_local(vs_path)
            return vs_path, [source]
        except Exception as e:
            logger.error(e)
            return None, [source]


    def add_content_to_knowledge(self, vs_path, content_ls: List[Text], source: Text, metadata_ls: List[Dict] = None) \
            -> Text:
        docs = []
        for content, metadata in zip(content_ls, metadata_ls):
            # 'source' key is uses for orginal metadata, SOURCE_DOC is used for distingush the intent and get answer for different intent
            metadata.update({'source': source, SOURCE_DOC: source})
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)

        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.add_documents(docs)
        torch_gc()
        vector_store.save_local(vs_path)
        return vs_path


    def load_vs(self, vs_path: Text) -> VectorStore:
        vector_store = load_vector_store(vs_path, self.embeddings)
        return vector_store

