"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: thread_utils.py
@time: 2023/8/30 22:53
"""

from threading import Thread


class ThreadWithReturnValue(Thread):
    """
    thread = Thread(target=retrieval_qa.run, args=[query])
    thread.start()

       Args:
           Thread (_type_): _description_
       """

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return