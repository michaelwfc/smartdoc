
import argparse
from configs.config import GRADIO_SERVER_PORT


def get_webui_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int,default=GRADIO_SERVER_PORT)
    parser.add_argument(
        "--share",
        # action="store_true",
        default=True,
        type=bool,
        help="Whether to generate a public, shareable link.",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue.",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    # parser.add_argument(
    #     "--add-chatgpt",
    #     action="store_true",
    #     help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    # )
    # parser.add_argument(
    #     "--add-claude",
    #     action="store_true",
    #     help="Add Anthropic's Claude models (claude-2, claude-instant-1)",
    # )
    # parser.add_argument(
    #     "--add-palm",
    #     action="store_true",
    #     help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    # )
    # parser.add_argument(
    #     "--gradio-auth-path",
    #     type=str,
    #     help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    #     default=None,
    # )
    return parser