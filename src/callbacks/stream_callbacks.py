"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: document_loader.py
@time: 2023/8/30 22:32
"""

from queue import Queue
from typing import TYPE_CHECKING, Optional, Any
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamingOutCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, timeout: Optional[float] = None, *args, **kwagrs) -> None:
        super().__init__(*args, **kwagrs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

        # sys.stdout.write(token)
        # sys.stdout.flush()
        self.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

        self.end()

    def put(self, token: str):
        """
               Recives tokens, decodes them, and prints them to stdout as soon as they form entire words.
               """

        self.on_finalized_text(token)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""

        # Flush the cache, if it exists
        # if len(self.token_cache) > 0:
        #     text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        #     printable_text = text[self.print_len :]
        #     self.token_cache = []
        #     self.print_len = 0
        # else:
        printable_text = ""
        # self.next_tokens_are_prompt = True
        self.on_finalized_text(text=printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""

        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)

        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value