import json
import logging
import logging.handlers
import os
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import List

import click
import fastapi
import gradio as gr
import jsonlines
import rich
import uvicorn
from misattribution.generation.client import ChatApi, Message, ChatMessages
from markdown import markdown
from pydantic import BaseModel, Field

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def setup_logger():
    logger = logging.getLogger("hyst")
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()

app = fastapi.FastAPI(debug=True)


class Request(BaseModel):
    text: str = ""
    question: str
    system: str = ""
    plain: bool = False
    stops: List[str] = Field(default_factory=lambda: ["Human:"])
    stream: bool = True
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    presence_penalty: float = 1.0


class JsonViewer:
    @staticmethod
    def to_str(json_object, level=0, parent=""):
        if not isinstance(json_object, (dict, list)):
            return str(json_object)

        head_level = "#" * (level + 1)
        parts = []
        if isinstance(json_object, dict):
            for key, val in json_object.items():
                pkey = parent + "." + key
                if isinstance(val, (dict, list)):
                    parts.append(JsonViewer.to_str(val, level + 1, pkey))
                else:
                    parts.append(
                        f"{head_level} {pkey}:\n\n{JsonViewer.to_str(val, level + 1, pkey)}"
                    )
        if isinstance(json_object, list):
            for i, item in enumerate(json_object):
                pkey = parent + "." + str(i)
                if isinstance(item, (dict, list)):
                    parts.append(JsonViewer.to_str(item, level + 1, pkey))
                else:
                    parts.append(
                        f"{head_level} {pkey}:\n\n{JsonViewer.to_str(item, level + 1, pkey)}"
                    )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def open_file(path):
        # reader = iter(open(path, "r", encoding="utf8"))
        reader = map(
            partial(json.dumps, ensure_ascii=False, indent=4),
            iter(jsonlines.open(path, "r")),
        )
        js = ""
        try:
            js = next(reader)
        except StopIteration:
            pass
        js, js, mk = JsonViewer.to_output(js)
        return js, js, mk, {"reader": reader, "line_no": 0, "js": js}, 0

    @staticmethod
    def to_output(js_str):
        js = json.loads(js_str)
        md = JsonViewer.to_str(js)
        mk = markdown(md, extensions=["tables"], output_format="html")
        return js_str, js_str, mk

    @staticmethod
    def from_js_str(js_str):
        return JsonViewer.to_output(js_str)

    @staticmethod
    def jump(path, line_no, state):
        if line_no == state["line_no"]:
            return state["js"], state["js"], state

        if line_no < state["line_no"]:
            js, _, state, _ = JsonViewer.open_file(path)
        i = state["line_no"]
        while i < line_no:
            try:
                js = next(state["reader"])
            except StopIteration:
                raise ValueError(f"超过行数 {i+1}")
            i += 1
        state["line_no"] = i
        js, js, mk = JsonViewer.to_output(js)
        return js, js, mk, state

    @staticmethod
    def demo():
        def load_str(text):
            try:
                t = json.loads(text)
            except json.decoder.JSONDecodeError:
                t = text
            print("str:", repr(t))
            mk = markdown(t, extensions=["tables"], output_format="html")
            return mk, t

        with gr.Blocks(analytics_enabled=False, title="工具集") as demo:
            state = gr.State(None)
            with gr.Tab("json viewer"):
                path = gr.Textbox(label="输入文件路径")
                line_no = gr.Number(value=0, label="跳到行", minimum=0, step=1)
                direct = gr.Textbox(label="输入json")
                with gr.Tab("视图1"):
                    json_viewer = gr.JSON()
                with gr.Tab("视图2"):
                    code_viewer = gr.Code(language="json")
                with gr.Tab("视图3"):
                    markdown_viewer = gr.HTML()

                direct.submit(
                    JsonViewer.from_js_str,
                    direct,
                    [json_viewer, code_viewer, markdown_viewer],
                )

                path.submit(
                    JsonViewer.open_file,
                    [path],
                    [json_viewer, code_viewer, markdown_viewer, state, line_no],
                )
                line_no.submit(
                    JsonViewer.jump,
                    [path, line_no, state],
                    [json_viewer, code_viewer, markdown_viewer, state],
                )
            with gr.Tab("markdown viewer"):
                markdown_html = gr.HTML(render=False)
                # markdown = gr.Markdown(label="markdown", render=False)
                markdown_it = gr.Textbox(label="转成markdown")
                textview = gr.Textbox()
                markdown_it.submit(load_str, markdown_it, [markdown_html, textview])
                markdown_html.render()
        return demo


@dataclass
class ChatState:
    history: List[List[str]]
    text_input: str
    visible: bool
    model: str
    stop: str

    @staticmethod
    def default():
        return ChatState([], "", True, "qwen2.5", "")


class Gradio:
    @staticmethod
    def history_to_messages(history):
        messages = []
        for hist in history:
            messages.append(Message(role="user", content=hist[0]))
            messages.append(Message(role="assistant", content=hist[1]))
        chat_messages = ChatMessages(messages[:-1])
        return chat_messages

    @staticmethod
    def add_user_input(message, hist):
        hist.append([message, ""])
        return hist

    @staticmethod
    def clear_history(hist, state, index):
        state[index].history = []
        return [], state

    @staticmethod
    def undo_history(hist, state, index):
        hist = hist[:-1]
        state[index].history = hist
        return hist, state

    @staticmethod
    def switch_visible(state, index):
        state[index].visible = not state[index].visible
        label = "折叠"
        if not state[index].visible:
            label = "展开"
        return state, label

    @staticmethod
    def chatbot_wrap(history, model, stop):
        message, _ = history[-1]
        prompt = Gradio.history_to_messages(history)

        stops = [s.strip() for s in stop.split("|||") if s.strip()]
        api = ChatApi(model, retry=1)
        start = time.time()
        content = ""
        for content in api.iter_chat(prompt, stops=stops):
            history[-1][1] = content
            yield history
        end = time.time()
        rich.print({"user": message, "assistant": content})
        rich.print(f"\n[bold red]inference ended in {end-start:.3f} seconds[/bold red]")

    @staticmethod
    def update_state(history, state, index):
        state[index].history = history
        state[index].visible = True
        return state

    def gradio_demo():
        model_list = list(["qwen2.5"])
        title = """<h1 align="center">Playground</h1>"""

        css = """.contain { display: flex; flex-direction: column; }
#chatbot { flex-grow: 1; max-height: 960px;}
# right_bottom {
    position:fixed;
    right:0;
    bottom:0;
}
# """

        demo = gr.Blocks(css=css, theme=gr.themes.Soft(), title="混元对话工具")
        with demo:
            gr.HTML(title)
            chatbots_state = gr.State([ChatState.default()])
            add_btn = gr.Button("增加对话", visible=False)
            add_btn.click(
                lambda state_list: state_list + [ChatState.default()],
                chatbots_state,
                chatbots_state,
            )

            @gr.render(inputs=[chatbots_state])
            def reader_chatbots(chat_state: List[ChatState]):
                print(chat_state)
                for i, state in enumerate(chat_state):
                    gr.HTML("<hr>")
                    visible = state.visible
                    index_state = gr.State(i)
                    with gr.Accordion("推理参数设置", open=False, visible=visible):
                        with gr.Row():
                            model = gr.Radio(
                                choices=model_list,
                                value=state.model,
                                label="模型",
                                visible=visible,
                            )

                            def radio_change(value, state, index):
                                state[index].model = value
                                return state

                            model.input(
                                radio_change,
                                [model, chatbots_state, index_state],
                                chatbots_state,
                            )

                            stop = gr.Textbox(
                                label="停止词,用|||分割多个",
                                value=state.stop,
                                visible=visible,
                            )

                            def stop_change(value, state, index):
                                state[index].stop = value
                                return state

                            stop.submit(
                                stop_change,
                                [stop, chatbots_state, index_state],
                                chatbots_state,
                            )
                    accordion = gr.Accordion(f"*对话流程{i + 1}*", open=visible)
                    with accordion:
                        chatbot = gr.Chatbot(
                            elem_id="chatbot", label="对话", value=state.history
                        )
                        message = gr.Textbox(
                            label="此处输入消息", value=state.text_input
                        )

                        def record_input(text, state, index):
                            state[index].text_input = text
                            return state

                        message.blur(
                            record_input,
                            [message, chatbots_state, index_state],
                            chatbots_state,
                        )
                    with gr.Row():
                        with accordion:
                            b3 = gr.Button("重新输入")
                            b2 = gr.Button("清除对话历史")
                            b4 = gr.Button("折叠")

                        b4.click(
                            Gradio.switch_visible,
                            [chatbots_state, index_state],
                            [chatbots_state, b4],
                        )
                        b3.click(
                            Gradio.clear_history,
                            [
                                chatbot,
                                chatbots_state,
                                index_state,
                            ],
                            [chatbot, chatbots_state],
                        ).then(
                            Gradio.add_user_input,
                            [message, chatbot],
                            [chatbot],
                        ).then(
                            Gradio.chatbot_wrap,
                            [chatbot, model, stop],
                            [chatbot],
                            concurrency_limit=32,
                        ).then(
                            Gradio.update_state,
                            [chatbot, chatbots_state, index_state],
                            chatbots_state,
                        )

                        message.submit(
                            Gradio.add_user_input,
                            [message, chatbot],
                            [chatbot],
                        ).then(
                            Gradio.chatbot_wrap,
                            [chatbot, model, stop],
                            [chatbot],
                            concurrency_limit=32,
                        ).then(
                            Gradio.update_state,
                            [chatbot, chatbots_state, index_state],
                            chatbots_state,
                        )

                        b2.click(
                            Gradio.clear_history,
                            [
                                chatbot,
                                chatbots_state,
                                index_state,
                            ],
                            [chatbot, chatbots_state],
                        )

                gr.HTML("<hr>")
                with gr.Row():
                    add_btn = gr.Button("增加对话")
                    del_btn = gr.Button("删除对话")

                    def pop(states):
                        states.pop()
                        return states

                    del_btn.click(pop, chatbots_state, chatbots_state)

                    def add(states):
                        states.append(ChatState.default())
                        return states

                    add_btn.click(
                        add,
                        [chatbots_state],
                        chatbots_state,
                    )

        return demo


@click.command()
@click.option("--port", default=8111, help="Number of greetings.")
def main(port):
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    demo = Gradio.gradio_demo()
    app = gr.mount_gradio_app(app, demo, path="/webui")
    app = gr.mount_gradio_app(app, JsonViewer.demo(), path="/jsonviewer")
    main()
