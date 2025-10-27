import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch

torch.set_grad_enabled(False)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor
from modelscope import snapshot_download
from peft import PeftModel
from qwen_vl_utils import process_vision_info


# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


def plot_images(image_paths):
    num_images = len(image_paths)

    # 创建图形并显示图片
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


import base64


def mllm_openai(query, images, conversation_history):
    model_path=os.path.join(os.path.dirname(os.getcwd()),'Qwen2.5-VL-Base-Answer')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    processor=Qwen2_5_VLProcessor.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
    base_model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
    base_model.to(device)
    base_model.eval()
    lora_path=snapshot_download(model_id='Johnnycreation/qlora-finetune-qwen2.5vl-3B',local_dir='../Qwen2.5-VL-Answer')
    lora_model=PeftModel.from_pretrained(base_model,lora_path)


    modified_conversation_history = [
        {"role": message["role"], "content": message["content"]}
        for message in conversation_history
    ]

    messages = [{"role": "system", "content": [{'type': 'text', 'text': "You are a helpful assistant."}]}]
    # messages = []
    messages.extend(modified_conversation_history)

    if len(images) != 0:
        messages.append({
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": query},
            ],
        })

        conversation_history.append({
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": query},
            ],
        })
        # print(messages)
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        conversation_history.append({"role": "user", "content": [{"type": "text", "text": query}]})

    #print("messages:",messages)
    texts=[processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)]
    #print(f"texts:{texts}")

    image_info,video_info=process_vision_info(messages)
    #print(f"image_info:{image_info},video_info:{video_info}")

    inputs=processor(text=texts,images=image_info,videos=video_info,padding=True,return_tensors="pt")
    inputs = inputs.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    # print(f'input-keys:{inputs.keys()}')

    generated_ids=lora_model.generate(**inputs,max_new_tokens=128)
    # print(generated_ids)
    generated_ids_trimmed=[output_ids[len(input_ids):] for input_ids,output_ids in zip(inputs.input_ids,generated_ids)]

    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    """
        response = processor.decode(
        generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    """
    # print(response)

    assistant_response = [dict(type='text',text=response[0])]
    conversation_history.append({"role": "assistant", "content": assistant_response})

    return assistant_response, conversation_history


# import requests


def image_formatting(image_path):
    image_path = image_path.strip('/')
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    return None


if __name__ == '__main__':
    from PIL import Image

    img = Image.open('mmdu_pics/2ab2903b15c2451680a95402aa58b93b.jpg')
    print(img)

    benchmark_path = os.path.join(os.path.dirname(os.getcwd()), 'benchmark.json')
    model_answer_save_path = os.path.join(os.path.dirname(os.getcwd()), 'qwen2.5-peft-gene')
    os.makedirs(model_answer_save_path, exist_ok=True)

    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmarks = json.load(f)

    print(len(benchmarks))
    print(type(benchmarks))

    print(benchmarks[0])
    print([bm['image'] for bm in benchmarks])

    import os

    # im2url = json.load(open('./scripts/cab357qiu.json'))
    for item in tqdm(benchmarks):
        record_data = item.copy()
        img_paths = item["image"]

        data_id = item["id"]
        file_path = f"{model_answer_save_path}/{data_id}.json"
        if os.path.exists(file_path):
            print(f"File exists: {file_path}, skipping this iteration.")
            continue  # 跳过该次循环

        ### 获取问题
        conv = item["conversations"]
        questions = []
        for i in conv:
            if i["from"] == "user":
                questions.append(i["value"])
        # print(questions)

        ### 遍历每一个问题
        pics_number = 0
        history = []
        try:
            for index, q in enumerate(questions):
                if "<ImageHere>" in q:
                    tag_number = q.count('<ImageHere>')
                    if tag_number == 1:
                        pics_number += 1
                        images = [img_paths[pics_number - 1]]
                    else:
                        pics_number_end = pics_number + tag_number
                        images = img_paths[pics_number: pics_number_end]
                        pics_number += tag_number
                else:
                    images = []

                images = [image_formatting(image) for image in images]

                # base64 form
                # print(images)
                with torch.cuda.amp.autocast():
                    response, history = mllm_openai(query=q, images=images, conversation_history=history)
                print(GREEN + response[0]["text"] + RESET)
                print(RED + q + RESET)


                record_data["conversations"][index * 2 + 1]["value"] = response

            with open(file_path, "w") as json_file:
                json.dump(record_data, json_file)
        except Exception as e:
            print({e})





