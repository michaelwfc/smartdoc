"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 20:10
"""
from typing import List,Dict,Text
from langchain.agents import BaseSingleActionAgent,BaseMultiActionAgent




class BaseAgent():
 def run(self,*args,**kwargs):
    raise NotImplementedError