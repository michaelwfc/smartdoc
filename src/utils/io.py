"""
@author: michaelwfc
@version: 1.0.0
@license: Apache Licence
@file: io.py
@description: 
@time: 2023/9/16 9:18
"""
import os


def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]


def get_file_type(filepath):
    if filepath.lower().endswith(".md"):
        file_type = "markdown"
    elif filepath.lower().endswith(".txt"):
        file_type = "text"
    elif filepath.lower().endswith(".pdf"):
        file_type = "pdf"
    elif filepath.lower().endswith(".jpg"):
        file_type = "jpg"
    elif filepath.lower().endswith(".png"):
        file_type = "png"
    elif filepath.lower().endswith(".csv"):
        file_type = "csv"
    else:
        file_type = "unknown"
    return file_type
