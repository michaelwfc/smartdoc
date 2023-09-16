#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :  webui.py
@Desc: idea from fastchat/serve/gradio_web_server.py
@Time: 7/13/2023 2:25 PM
@Contact :
@License :   Create on 7/13/2023, Copyright 2023  All rights reserved.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
7/13/2023 2:25 PM         1.0

'''
import os
import shutil
import sys
import traceback
import time
from typing import Text, List
import json
import requests
from utils.loggers import init_logger
from vectorstores.vs_builders import VectorStoreBuilder

logger = init_logger()

import gradio as gr
from gradio import State as gr_State
from fastchat.conversation import Conversation
# from fastchat.serve.gradio_web_server
from servers.gradio_web_server import block_css, get_window_url_params_js, ip_expiration_dict, \
    SESSION_EXPIRATION_TIME, \
    upvote_last_response, downvote_last_response, flag_last_response, regenerate, clear_history, \
    get_conv_log_filename, post_process_code, \
    no_change_btn, State, violates_moderation, MODERATION_MSG, CONVERSATION_TURN_LIMIT, INACTIVE_MSG, \
    CONVERSATION_LIMIT_MSG, \
    INPUT_CHAR_LEN_LIMIT, SERVER_ERROR_MSG, ErrorCode

from constants import PROJECT_NAME, GPT_3DOT5_TURBO
from configs.config import USE_AZURE_OPENAI, USE_VICUNA,  SUPPORT_MODELS, KB_ROOT_PATH, TMP_FILES_DIRNAME
from utils.environments import set_gpt_env
from utils.argument_parser import get_webui_args_parser

# from utils.database import get_snowflake_uri
# from agents.router_agents import TopicRouterAgent

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

controller_url = None
enable_moderation = False


def build_demo(models, args):
    """
    The Blocks app code will be contained within this clause.

    :param models:
    :param args:
    :return:
    """
    with gr.Blocks(
            title="AIFE Bot",
            # Gradio themes are the easiest way to customize the look and feel of your app: gr.themes.Base()
            theme=gr.themes.Soft(),
            css=block_css) as demo:
        url_params = gr.JSON(visible=False)

        # build single model ui
        # add vs_path_state, file_status_state, model_status_state,
        (vs_path_state, file_status_state, model_status_state,
         state,
         model_selector,
         chatbot,
         textbox,
         send_btn,
         button_row,
         parameter_row,
         doc_uploader_row
         ) = build_single_model_ui(models)

        demo.load(
            load_demo,
            [url_params],
            # add vs_path_state, file_status_state, model_status_state,
            [vs_path_state, file_status_state, model_status_state,
             state,
             model_selector,
             chatbot,
             textbox,
             send_btn,
             button_row,
             parameter_row,
             doc_uploader_row
             ],
            _js=get_window_url_params_js,
        )

    return demo


def load_demo(url_params, request: gr.Request):
    global models
    global args
    ip = request.client.host
    logger.info(f"load demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME
    return load_demo_single(models, url_params)


def load_demo_single(models, url_params):
    """
    ValueError: An event handler (load_demo) didn't receive enough output values (needed: 8, received: 7).

    :param models:
    :param url_params:
    :return:
    """
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )
    # add vs_path_state, file_status_state, model_status_state
    vs_path_state, file_status_state, model_status_state = gr.State(""), gr.State(""), gr.State("")

    state = None
    return (
        vs_path_state, file_status_state, model_status_state,
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
        # gr.Audio.update(visible=True)  # add
        # gr.File.update(visible=True),
        gr.File.update(visible=True)
    )


def build_single_model_ui(models: List[Text], add_promotion_links=False, topic_anget=None):
    vs_path_state, file_status_state, model_status_state = gr.State(""), gr.State(""), gr.State("")

    # state, where data persists across multiple submits within a page session, in Blocks apps as well.

    state = gr.State()
    # define the PROJECT_NAME
    notice_markdown = f"""# {PROJECT_NAME}"""
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Row():
        with gr.Column(scale=15):
            # build the chatbot
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Scroll down and start chatting",
                visible=False,
                height=550,

            )
        # build the model_selector_row, set visible= False
        with gr.Column(elem_id="model_input_column", visible=True, scale=1):
            with gr.Row(elem_id="model_selector_row", visible=True):
                model_selector = gr.Dropdown(
                    choices=models,
                    value=models[0] if len(models) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                )

            # TODO: add doc_uploader_row
            with gr.Row(elem_id="upload_document", visible=False) as doc_uploader_row:
                files_input = gr.File(label="upload document here",
                                      file_types=[".csv", '.txt', '.md', '.pdf'],
                                      file_count="multiple",
                                      show_label=True,
                                      visible=True)

            # build Parameters Accordion
            with gr.Accordion("Parameters", open=True, visible=True) as parameter_row:
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Top P",
                )
                max_output_tokens = gr.Slider(
                    minimum=16,
                    maximum=1024,
                    value=512,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )
    # change event: input the files and update the vs_path_state
    files_input.change(build_vectore_store,
                       inputs=[vs_path_state, file_status_state, chatbot, model_selector, files_input],
                       outputs=[vs_path_state, file_status_state, chatbot], show_progress=True)


    # build the textbox and send botton
    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    # build the button row
    with gr.Row(visible=False) as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    # remove the markdown
    # gr.Markdown(learn_more_md)

    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

    # Register listeners for upvote button
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )

    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, vs_path_state, chatbot, model_selector, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [
        state, chatbot, textbox] + btn_list)

    #  add_text to state in textbox submit
    textbox.submit(
        add_text, [state, model_selector, textbox], [
                                                        state, chatbot, textbox] + btn_list
    ).then(
        # call response to state
        bot_response,
        [state, vs_path_state, chatbot, model_selector, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    send_btn.click(
        add_text, [state, model_selector, textbox], [
                                                        state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, vs_path_state, chatbot, model_selector, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    # add load_file_button event
    # load_file_button.click(get_vector_store,
    #                        show_progress=True,
    #                        inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
    #                        outputs=[vs_path, files, chatbot, files_to_delete], )

    # add ars model
    # whisper_pipeline = load_whiper_pipeline()
    # # partial_transcribe = partial(transcribe_audio, model=whisper_pipeline)
    # transcribe_audio =  transcribe_audio_outer(model=whisper_pipeline)

    # with gr.Accordion("Transcribe Audio", open=False, visible=False) as audio_transcriber:
    #     # with gr.Row():
    #     with gr.Column(scale=20):
    #         audio_input = Audio(source="upload", type="filepath")

    # audio_input.change(transcribe_audio, [state, model_selector, audio_input], [state, chatbot, textbox] + btn_list)

    return vs_path_state, file_status_state, model_status_state, \
        state, model_selector, chatbot, textbox, send_btn, button_row, parameter_row, doc_uploader_row


def add_text(state: gr_State, model_selector: Text, text: Text, request: gr.Request):
    ip = request.client.host
    logger.info(f"add text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    if ip_expiration_dict[ip] < time.time():
        logger.info(f"inactive. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), INACTIVE_MSG) + (no_change_btn,) * 5

    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(
                f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), MODERATION_MSG) + (
                no_change_btn,
            ) * 5

    # from fastchat.conversation import Conversation
    conv: Conversation = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(
            f"conversation turn limit. ip: {request.client.host}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (
            no_change_btn,
        ) * 5

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def build_vectore_store(vs_path_state: gr_State, file_status_state: gr_State, chatbot: gr.Chatbot,
                        model_selector: Text, files: List[Text], request: gr.Request):
    try:
        vs_builder = VectorStoreBuilder(use_openai=USE_OPENAI,use_vicuna=USE_VICUNA, use_azure_openai=USE_AZURE_OPENAI)
        if not isinstance(files, list):
            files = [files]
        filelist = []
        temp_file_dir = os.path.join(KB_ROOT_PATH, TMP_FILES_DIRNAME)
        if not os.path.exists(temp_file_dir):
            os.makedirs(temp_file_dir)
        for file in files:
            filename = os.path.split(file.name)[-1]
            shutil.move(file.name, os.path.join(temp_file_dir, filename))
            filelist.append(os.path.join(temp_file_dir, filename))

        vs_path, loaded_files = vs_builder.init_knowledge_vector_store(filepath=filelist)
        file_status = "<font color=Blue>**Success to load file!**</font>"

    except Exception as e:
        vs_path = ""
        file_status = f"<font color=Red>**Fail to load file,please reload the file! **</font>\nException:{e}"
    history = chatbot + [[None, file_status]]
    # update the component
    # return gr.update(value=vs_path),gr.update(value=file_status),history
    return vs_path, file_status, history


def bot_response(state: State, vs_path_state: gr.State, chat_history: gr.Chatbot,
                 model_selector: gr.Dropdown,
                 temperature, top_p, max_new_tokens, request: gr.Request):
    # from agents.openai_agents import OpenaiAgent
    from agents.doc_agents import DocAgent

    logger.info(f"bot response on ip: {request.client.host}")
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return
    conv, model_name = state.conv, state.model_name

    prompt = conv.to_openai_api_messages()
    # implent the chatdoc agent and add history
    agent = DocAgent(streaming=True,use_openai=USE_OPENAI,
                     use_azure_openai=USE_AZURE_OPENAI, use_vicuna=USE_VICUNA,
                     vs_path=vs_path_state)
    query = prompt[-1]["content"]
    stream_iter = agent.run(query=query, vs_path=vs_path_state, chat_history=chat_history, model=model_selector)

    conv.update_last_message("▌")
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5


    try:
        for data in stream_iter:
            if data["error_code"] == 0:
                output = data["text"].strip()
                if "vicuna" in model_name:
                    output = post_process_code(output)
                conv.update_last_message(output + "▌")
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + \
                         f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
            time.sleep(0.015)
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})")
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn,)
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})")
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn,)
        return

    # Delete "▌"
    conv.update_last_message(conv.messages[-1][-1][:-1])
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def run_chat_server(use_openai=True, use_azure_openai=False, use_vicuna=False):
    global models
    global args

    # set env
    set_gpt_env(use_openai=use_openai, use_azure_openai=use_azure_openai, use_vicuna=use_vicuna)
    # set the args parser

    parser = get_webui_args_parser()
    args = parser.parse_args()
    logger.info(f"args: {args.__repr__()}")

    models = SUPPORT_MODELS
    # # Set authorization credentials

    auth = None

    # Launch the demo
    demo = build_demo(models, args)

    # demo.launch()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=20,
        auth=auth,
    )


if __name__ == '__main__':
    try:
        logger.info("run chat webui")
        from configs.config import  USE_OPENAI
        run_chat_server(use_openai = USE_OPENAI)
    except Exception as e:
        print(traceback.format_exc())
    # logger.error(e,exc_info=True)
    sys.exit(0)
