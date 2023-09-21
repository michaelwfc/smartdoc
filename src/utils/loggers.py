"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: loggers.py
@description: 
@time: 2023/9/16 8:42
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 15:45 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 15:45   wangfc      1.0         None
"""

import os
import logging

from typing import Optional, Text
import time
from pathlib import Path
from constants import DEFAULT_ENCODING
from utils.time_utils import TIME_FORMAT

DEFAULT_LOG_LEVEL = "INFO"
ENV_LOG_LEVEL = "LOG_LEVEL"
LOG_FORMAT = '%(asctime)s[%(levelname)s]%(name)s:%(funcName)s[line:%(lineno)d]%(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def init_logger(log_filename='log', output_dir='logs',
                name=None,
                log_file_path=None,
                log_level=DEFAULT_LOG_LEVEL,
                fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT):
    '''
       Example:
       >>> init_logger(log_file)
       >>> logger.info("abc'")
    '''
    if isinstance(log_filename, Path):
        log_filename = str(log_filename)

    if output_dir and log_filename:
        timestamp = time.strftime(TIME_FORMAT, time.localtime())
        log_filename = f'{log_filename}_{timestamp}.log'
    log_file_path = os.path.join(str(Path(__file__).parent.parent.parent), output_dir, log_filename)

    log_level = get_log_level(log_level=log_level)
    log_format = get_log_format(fmt=fmt, datefmt=datefmt)

    logger = logging.getLogger(name)


    # 增加 console_handler:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # 增加 file_handler
    if log_file_path and log_file_path != '':
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding=DEFAULT_ENCODING)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.setLevel(log_level)

    return logger


def set_log_level(log_level: Optional[int] = None) -> None:
    """Set log level of Rasa and Tensorflow either to the provided log level or
       to the log level specified in the environment variable 'LOG_LEVEL'. If none is set
       a default log level will be used."""

    if not log_level:
        log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        log_level = logging.getLevelName(log_level)

    logging.getLogger("faq").setLevel(log_level)

    # update_tensorflow_log_level()
    # update_asyncio_log_level()
    # update_matplotlib_log_level()
    # update_apscheduler_log_level()
    # update_socketio_log_level()

    os.environ[ENV_LOG_LEVEL] = logging.getLevelName(log_level)


def get_log_level(log_level: Text = 'info'):
    if log_level is None:
        log_level = DEFAULT_LOG_LEVEL

    if isinstance(log_level, str):
        if log_level.lower() == 'debug':
            log_level = logging.DEBUG
        elif log_level.lower() == 'info':
            log_level = logging.INFO
        elif log_level.lower() == 'warn':
            log_level = logging.WARNING
    return log_level


def get_log_format(fmt=LOG_FORMAT, datefmt=None):
    log_format = logging.Formatter(fmt=fmt, datefmt=datefmt)
    return log_format


def is_logging_disabled() -> bool:
    """Returns `True` if log level is set to WARNING or ERROR, `False` otherwise."""
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
    return log_level in ("ERROR", "WARNING")
