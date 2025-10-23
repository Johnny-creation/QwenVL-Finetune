import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info
import gradio as gr

warnings.filterwarnings("ignore")


def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def bot_streaming(message, history, generation_args):
    """多轮对话 + 多模态支持（与你的版本完全一致）"""
    images, videos = [], []
    if message.get("files"):
        for file_item in message["files"]:
            file_path = file_item["path"] if isinstance(file_item, dict) else file_item
            if is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    conversation = []
    conversation.append({
        "role": "system",
        "content": [{"type": "text", "text": "你是一个多模态智能助手，请记住之前的对话内容并连续回答。"}]
    })

    for user_turn, assistant_turn in history:
        if user_turn is None and assistant_turn is None:
            continue

        user_content = []

        if isinstance(user_turn, tuple):
            file_paths = []
            user_text = ""
            if len(user_turn) >= 1:
                f0 = user_turn[0]
                if f0:
                    file_paths = f0 if isinstance(f0, list) else [f0]
            if len(user_turn) >= 2 and user_turn[1] is not None:
                user_text = user_turn[1]

            for file_path in file_paths:
                if is_video_file(file_path):
                    user_content.append({"type": "video", "video": file_path, "fps": 1.0})
                else:
                    user_content.append({"type": "image", "image": file_path})

            if user_text:
                user_content.append({"type": "text", "text": user_text})

        elif isinstance(user_turn, str):
            user_content.append({"type": "text", "text": user_turn})

        if user_content:
            conversation.append({"role": "user", "content": user_content})

        if assistant_turn:
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_turn}]
            })

    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps": 1.0})
    if message.get('text'):
        user_content.append({"type": "text", "text": message['text']})
    if user_content:
        conversation.append({"role": "user", "content": user_content})

    conversation = conversation[-40:]

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
        clean_up_tokenization_spaces=False,
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=generation_args.get("max_new_tokens", 1024),
        do_sample=True if generation_args.get("temperature", 0) > 0 else False,
        temperature=generation_args.get("temperature", 0) if generation_args.get("temperature", 0) > 0 else None,
        repetition_penalty=generation_args.get("repetition_penalty", 1.0),
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer


def main(args):
    global processor, model, device

    device = args.device
    disable_torch_init()

    use_flash_attn = True
    model_name = get_model_name_from_path(args.model_path)
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=args.device,
        model_name=model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=args.device,
        use_flash_attn=use_flash_attn
    )

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args)

    # ✨ 改造后的 UI ✨
    with gr.Blocks(
        title="县域土地利用智能问答平台",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
        css=".gradio-container {background: #f7fafc;}"
    ) as demo:

        # 顶部标题区
        with gr.Column(scale=1, elem_id="app_header"):
            gr.Markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #1f2937; font-size: 2.5em; font-weight: 600;">🛰️ 县域土地利用智能问答平台</h1>
                <p style="color: #4b5563; font-size: 1.1em;">
                    面向中西部县域自然资源管理的分钟级动态监测与决策支持系统
                </p>
                <p style="color: #6b7280; font-size: 0.9em; margin-top: 8px;">
                    基于低分辨率卫星影像和大模型问答技术，解决“数据难获取、分析门槛高、监测不及时”等痛点，让非专业人员也能通过自然语言提问快速获取土地动态信息。
                </p>
            </div>
            """)

        # 功能价值卡片
        with gr.Row(equal_height=True, variant="panel", elem_id="feature_cards"):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("""
                <div style="padding: 15px; border-radius: 12px; background: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); text-align: center;">
                    <h3 style="color: #16a34a;">📊 数据适配性强</h3>
                    <p style="color: #374151;">支持多源低分辨率卫星影像，自动匹配县域尺度数据</p>
                </div>
                """)
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("""
                <div style="padding: 15px; border-radius: 12px; background: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); text-align: center;">
                    <h3 style="color: #2563eb;">💡 问答贴近业务</h3>
                    <p style="color: #374151;">用自然语言提问即可，如“新增耕地面积”“疑似违规建设”</p>
                </div>
                """)
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("""
                <div style="padding: 15px; border-radius: 12px; background: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); text-align: center;">
                    <h3 style="color: #f97316;">⚡ 分钟级响应</h3>
                    <p style="color: #374151;">模型生成时间 < 2 分钟，助力审批、执法与报表决策</p>
                </div>
                """)

        # 主功能交互区
        with gr.Column(elem_id="main_chat_interface"):
            gr.Markdown("<h2 style='text-align: center; margin-top: 25px; margin-bottom: 20px; color: #1f2937;'>🔍 智能问答交互面板</h2>")

            chatbot = gr.Chatbot(
                height=550,
                label="智能对话记录",
                bubble_full_width=False,
                avatar_images=(None, "https://img.alicdn.com/imgextra/i3/O1CN01eJSPaL1UXrmpANQjB_!!6000000002540-2-tps-128-128.png") # 添加一个AI头像
            )
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_types=["image", "video"],
                placeholder="请输入问题或上传影像文件（如：'2024年上半年耕地面积变化是多少？'）",
                show_label=False,
                file_count="multiple"
            )

            gr.ChatInterface(
                fn=bot_streaming_with_args,
                title=None, # 隐藏默认标题
                stop_btn="⏹️ 停止生成",
                chatbot=chatbot,
                textbox=chat_input,
                examples=[
                    "2025 年新增耕地面积是多少？",
                    "最近一个季度是否存在违规建设？",
                    "2024 与 2025 年土地利用结构有何变化？"
                ],
            )

        # 快速提问推荐
        with gr.Accordion("📌 快速提问示例", open=False):
            gr.Markdown("""
            - 📈 本季度耕地面积变化情况？
            - 🏗️ 疑似违规建设的分布区域？
            - 🌿 保护区边界是否被占用？
            - 🗺️ 近一年土地利用结构趋势？
            """)

    demo.queue(api_open=False)
    demo.launch(show_api=False, share=False, server_name='0.0.0.0')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
