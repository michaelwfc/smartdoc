"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: doc_agent_templates.py
@time: 2023/8/30 22:45
"""
CONTEXT_QA_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""