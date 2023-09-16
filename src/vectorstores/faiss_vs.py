"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: faiss_vs.py
@time: 2023/8/30 22:58
"""

from langchain.vectorstores import FAISS, DocArrayInMemorySearch
from langchain.vectorstores.base import VectorStore, VST
from typing import Any, Callable, List, Dict, Optional, Type, Text
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import numpy as np
import os
from configs.config import VECTOR_SEARCH_SCORE_THRESHOLD, CHUNK_SIZE
from constants import OPENAI_EMBEDDING_MAX_INPUT_NUM


class BatchDocArrayInMemorySearch(DocArrayInMemorySearch):
    @classmethod
    def from_text_with_batch(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[Dict[Any, Any]]] = None,
            max_input_num=OPENAI_EMBEDDING_MAX_INPUT_NUM,
            **kwargs: Any,
    ) -> DocArrayInMemorySearch:
        """Create an DocArrayInMemorySearch store and insert data.
               Args:
                   texts (List[str]): Text data.
                   embedding (Embeddings): Embedding function.
                   metadatas (Optional[List[Dict[Any, Any]]]): Metadata for each text
                       if it exists. Defaults to None.
                   metric (str): metric for exact nearest-neighbor search.
                       Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".
                       Defaults to "cosine_sim".
               Returns:
                   DocArrayInMemorySearch Vector Store
               """

        store = cls.from_params(embedding, **kwargs)
        batch_indeies = np.arange(0, texts.__len__(), max_input_num)
        for start_batch_index in batch_indeies:
            end_batch_index = start_batch_index + max_input_num
            end_batch_index = texts.__len__() if end_batch_index > texts.__len__() else end_batch_index
            batch_texts = texts[start_batch_index:end_batch_index]
            batch_metadata = metadatas[start_batch_index:end_batch_index] if metadatas else None
            store.add_texts(texts=batch_texts, metadatas=batch_metadata)
            print(f"add_text  index from {start_batch_index} to {end_batch_index}")
        return store


class MyFAISS(FAISS, VectorStore):
    def __init__(
            self,
            embedding_function: Callable,
            index: Any,
            docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
            normalize_L2: bool = False,
    ):
        super().__init__(embedding_function=embedding_function,
                         index=index,
                         docstore=docstore,
                         index_to_docstore_id=index_to_docstore_id,
                         normalize_L2=normalize_L2)

        self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD
        self.chunk_size = CHUNK_SIZE
        self.chunk_conent = False

    def seperate_list(self, ls: List[int]) -> List[List[int]]:
        # TODO: 增加是否属于同一文档的判断
        lists = []
        ls1 = [ls[0]]
        for i in range(1, len(ls)):
            if ls[i - 1] + 1 == ls[i]:
                ls1.append(ls[i])
            else:
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists


    @classmethod
    def from_documents_with_batch(
            cls: Type[VST],
            documents: List[Document],
            embedding: Embeddings,
            max_input_num=OPENAI_EMBEDDING_MAX_INPUT_NUM,
            **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        batch_indeies = np.arange(0, texts.__len__(), max_input_num)
        if batch_indeies.__len__() == 1:
            vs = cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
        else:
            start_batch_index = batch_indeies[0]
            end_batch_index = start_batch_index + max_input_num
            initial_batch_texts = texts[start_batch_index:end_batch_index]
            initial_batch_metadata = metadatas[start_batch_index:end_batch_index]
            vs = cls.from_texts(initial_batch_texts, embedding, metadatas=initial_batch_metadata, **kwargs)
        for start_batch_index in batch_indeies[1:]:
            end_batch_index = start_batch_index + max_input_num
            end_batch_index = texts.__len__() if end_batch_index > texts.__len__() else end_batch_index

            batch_texts = texts[start_batch_index:end_batch_index]
            batch_metadata = metadatas[start_batch_index:end_batch_index] if metadatas else None
            vs.add_texts(texts=batch_texts, metadatas=batch_metadata, **kwargs)
            # store.add_texts(texts=batch_texts, metadatas=batch_metadata)
            print(f"add_text  index from {start_batch_index} to {end_batch_index}")

        # return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
        return vs


    def delete_doc(self, source: str or List[str]):
        try:
            if isinstance(source, str):
                ids = [k for k, v in self.docstore._dict.items(
                ) if v.metadata["source"] == source]
                vs_path = os.path.join(os.path.split(
                    os.path.split(source)[0])[0], "vector_store")
            else:
                ids = [k for k, v in self.docstore._dict.items(
                ) if v.metadata["source"] in source]
                vs_path = os.path.join(os.path.split(
                    os.path.split(source[0])[0])[0], "vector_store")
            if len(ids) == 0:
                return f"docs delete fail"
            else:
                for id in ids:
                    index = list(self.index_to_docstore_id.keys())[
                        list(self.index_to_docstore_id.values()).index(id)]
            self.index_to_docstore_id.pop(index)
            self.docstore._dict.pop(id)
            # TODO: 从 self.index 中删除对应id
            # self.index.reset()
            self.save_local(vs_path)
            return f"docs delete success"
        except Exception as e:
            print(e)
            return f"docs delete fail"


    def update_doc(self, source, new_docs):
        try:
            delete_len = self.delete_doc(source)
            ls = self.add_documents(new_docs)
            return f"docs update success"
        except Exception as e:
            print(e)
            return f"docs update fail"


    def list_source(self) -> List[Text]:
        return list(set(v.metadata["source"] for v in self.docstore._dict.values()))


    def list_docs(self) -> List[Document]:
        return list(self.docstore._dict.values())