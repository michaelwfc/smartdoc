"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 22:32

streaming:
1. use callback
2. add streamer to DocRetrievalQA
Reference:
https://github.com/langchain-ai/langchain/issues/4950
https://medium.com/@shrinath.suresh/building-an-interactive-streaming-chatbot-with-langchain-transformers-and-gradio-93b97378353e
custom .stream() for llm
"""

import inspect
from typing import Dict, Tuple, List, Any, Optional, Union
from langchain.callbacks.manager import Callbacks, CallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.load.dump import dumpd
from langchain.schema import RUN_KEY, RunInfo
from langchain.docstore.document import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


class DocRetrievalQA(RetrievalQA):
    """
       BaseRetrievalQA.run()
           -> .___call__() -> ._call()
           docs = self._get_docs()
           answer = self.combine_documents_chain.run(docs, question, callbacks)
       StuffDocumentsChain.run() :Any
           ->.combine_docs():Tuple[str, dict] -> .llm_chain.predict() : str

       LLM_Chain.predict(): str
           ->._call(): Dict[str, str] -> .generate():LLMResult -> .self.llm.generate_prompt():LLMResult

       LLM.generate_prompt(): LLMResult
            -> .generate():LLMResult -> ._generate_with_cache(): ChatResult  ->_generate(): ChatResult ->_stream():ChatGenerationChunk

       Args:
           RetrievalQA (_type_): _description_
       """

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
            ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

               If chain has 'return_source_documents' as 'True', returns
               the retrieved documents as well under the key 'source_documents'.

               Example:
               .. code-block:: python

               res = indexqa({'query': 'This is my query'})
               answer, docs = res['result'], res['source_documents']
               """

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
                "run_manager" in inspect.signature(self._get_docs).parameters
        )


        if accepts_run_manager:
            docs = self._get_docs(question, run_manager=_run_manager)
        else:
            docs = self._get_docs(question)  # type: ignore[call-arg]

        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


    def call_with_docs(self,
                       inputs: Union[Dict[str, Any], Any],
                       docs: List[Document],
                       return_only_outputs: bool = False,
                       callbacks: Callbacks = None,
                       *,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       include_run_info: bool = False,
                       ) -> Dict[str, Any]:
        inputs = self.prep_inputs(inputs)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
        )
        try:
            outputs = (
                self._call_with_docs(inputs, docs, run_manager=run_manager, )
                if new_arg_supported
                else self._call(inputs)
            )
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e
        run_manager.on_chain_end(outputs)
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs


    def _call_with_docs(self,
                        inputs: Dict[str, Any], docs=List[Document],
                        run_manager: Optional[CallbackManagerForChainRun] = None,
                        ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        accepts_run_manager = (
                "run_manager" in inspect.signature(self._get_docs).parameters
        )

        # get the answer with .generate() method
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


    def get_source_doc_info(self,
                            inputs: Union[Dict[str, Any], Any],
                            return_only_outputs: bool = False,
                            callbacks: Callbacks = None,
                            *,
                            tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None,
                            include_run_info: bool = False,
                            ) -> Dict[str, Any]:

        inputs = self.prep_inputs(inputs)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
        )
        try:
            outputs = (
                self._get_docs_info(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e
        run_manager.on_chain_end(outputs)
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs


    def _get_docs_info(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        accepts_run_manager = (
                "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(question, run_manager=_run_manager)
        else:
            docs = self._get_docs(question)  # type: ignore[call-arg]

        answer = None

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}