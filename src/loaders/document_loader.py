"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 22:32
"""
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from typing import List,Text
from langchain.document_loaders import UnstructuredMarkdownLoader, UnstructuredFileLoader, TextLoader, CSVLoader, \
    PyPDFLoader, UnstructuredImageLoader
from langchain.docstore.document import Document
# from loaders.pdf_loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from textsplitters.textspliter_utils import get_text_splitter
from textsplitters.zh_title_enhance import zh_title_enhance
# from textsplitters import ChineseTextSplitter
from configs.config import SENTENCE_SIZE,ZH_TITLE_ENHANCE
from constants import ENGLISH,CHINESE,SOURCE_DOC


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
            fout.close()



def load_langchain_docs(filepath, language=ENGLISH,
                        sentence_size=SENTENCE_SIZE, using_zh_title_enhance=ZH_TITLE_ENHANCE)-> List[Document]:
    if filepath.lower().endswith(".md"):
        docs= read_markdown_to_docs(filepath=filepath,mode="elements")
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = get_text_splitter(language=language, pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        # if language==CHINESE:
            #     loader = UnstructuredPaddlePDFLoader(filepath)
            # else:
        loader = PyPDFLoader(filepath)
        textsplitter = get_text_splitter(language=language,pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        # loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        # textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        loader = UnstructuredImageLoader(filepath,mode="elements")
        textsplitter = get_text_splitter(language=language, pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = get_text_splitter(language=language,pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    if using_zh_title_enhance:
        docs = zh_title_enhance(docs)
    # write_check_file(filepath, docs)
    return docs



def read_markdown_to_docs(filepath,mode="elements",**kwargs)-> List[Document]:
    loader = UnstructuredMarkdownLoader(filepath, mode=mode,**kwargs)
    docs = loader.load()
    return docs


def add_source_to_metadata(doc:Document,source:Text)->Document:
    """add SOURCE_DOC to metadata to track the source of doc"""
    metadata = doc.metadata
    metadata= {} if metadata is None else metadata
    metadata.update({SOURCE_DOC:source})
    return doc