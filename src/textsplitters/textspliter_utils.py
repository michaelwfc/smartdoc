"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: utils.py
@description:
@time: 2023/9/16 9:24
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from textsplitters.chinese_text_splitter import ChineseTextSplitter

from configs.config import SENTENCE_SIZE
from constants import ENGLISH, CHINESE


def get_text_splitter(pdf=False, sentence_size=SENTENCE_SIZE, language=ENGLISH):
    if language == CHINESE:
        textsplitter = ChineseTextSplitter(pdf=pdf, sentence_size=sentence_size)
    else:
        textsplitter = build_recursive_char_text_splitter()
    return textsplitter


def build_recursive_char_text_splitter(separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                                       chunk_size=1000,
                                       chunk_overlap=150,
                                       length_function=len
                                       ):
    r_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                             separators=separators)
    # c_splitter =  CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,
    #                                     separators =separators)


    return r_split
