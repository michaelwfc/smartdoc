"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: vs_utils.py
@time: 2023/8/30 23:17
"""
import os
from functools import lru_cache
import openai
from langchain.embeddings.base import Embeddings
from langchain.embeddings import  OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from vectorstores.faiss_vs import MyFAISS
from configs.config import embedding_model_dict,CACHED_VS_NUM,VICUNA_API_BASE,VICUNA_API_KEY
from constants import FAISS, DOCARRAY_INMEMORY_SEARCH,GPT_3DOT5_TURBO,DEPLOYMENT_NAME,OPENAI_API_KEY,OPENAI_API_BASE,AZURE,OPENAI_API_VERSION


# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    model_name = self.model_name  if hasattr(self,"model_name") else self.model
    return  hash(model_name)

HuggingFaceEmbeddings.__hash__ = _embeddings_hash
OpenAIEmbeddings.__hash__ = _embeddings_hash



def load_langchain_embeddings(use_openai = True, use_azure_openai=False,use_vicuna=False,
                              embedding_model=None,embedding_device=None)-> Embeddings:
    # use the sampe embeddings for vicuna and openai
    if use_azure_openai:
        # when use azure_openai, you should set all the params
        embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002",
        openai_api_type= AZURE,
        openai_api_key= os.getenv(OPENAI_API_KEY),
        openai_api_base=os.getenv(OPENAI_API_BASE),
        openai_api_version=os.getenv(OPENAI_API_VERSION) )
    elif use_openai or use_vicuna:
        # when use vicuna,set
        # os.environ[OPENAI_API_KEY]  = VICUNA_API_KEY
        # os.environ[OPENAI_API_BASE] =  VICUNA_API_BASE
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
        model_kwargs={'device': embedding_device})
        # sentence1 = "i like dogs"
        # embedding1 = embedding.embed_query(sentence1)
    return embeddings



# will keep CACHED_VS_NUM of vector store caches
@lru_cache(CACHED_VS_NUM)
def load_vector_store(vs_path=None, embeddings=None, docs=None,vs_type= FAISS):
    if vs_type == FAISS:
        vector_db =  MyFAISS.load_local(vs_path, embeddings)
    elif vs_type == DOCARRAY_INMEMORY_SEARCH:
        vector_db = DocArrayInMemorySearch.from_documents(
            docs,
            embeddings
        )
    # Similarity Search
        # related_docs = db.similarity_search(query,k=3)
    return vector_db