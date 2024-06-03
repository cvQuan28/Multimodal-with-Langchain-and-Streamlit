import os.path
from PIL import Image
from io import BytesIO

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
from utils import load_config
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# import streamlit as st
config = load_config()

model_image_path = config["llava_model"]["llava_model_path"]


def convert_bytes_to_base64(image_bytes):
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_string


def base64_to_image(base64_string):
    # Giải mã chuỗi base64 để nhận được dữ liệu nhị phân của hình ảnh
    image_data = base64.b64decode(base64_string)

    # Sử dụng PIL để mở hình ảnh từ dữ liệu nhị phân
    image = Image.open(BytesIO(image_data))

    return image


def load_checkpoint():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["lm_head", "embed_tokens"],
    )
    processor = AutoProcessor.from_pretrained(config["llava_model"]["llava_model_path"], use_auth_token=True)
    # Simply take-off the quantization_config arg if you want to load the original model
    model = IdeficsForVisionText2Text.from_pretrained(config["llava_model"]["llava_model_path"],
                                                      quantization_config=bnb_config, device_map="auto")

    if config["llava_model"]["checkpoint_path"]:
        print('Load checkpoint: ', config["llava_model"]["checkpoint_path"])
        peft_model = PeftModel.from_pretrained(model
                                               , config["llava_model"]["checkpoint_path"],
                                               is_trainable=False)
        return [peft_model, processor]
    return [model, processor]


# @st.cache_resource # can be cached if you use it often
def load_llava():
    if os.path.basename(model_image_path).split('.')[-1] == 'gguf':
        chat_handler = Llava15ChatHandler(clip_model_path=config["llava_model"]["clip_model_path"])
        llm = Llama(
            model_path=config["llava_model"]["llava_model_path"],
            chat_handler=chat_handler,
            logits_all=True,
            config=config["ctransformers"]["model_config"],
            n_ctx=1024  # n_ctx should be increased to accomodate the image embedding
        )
    else:
        llm = load_checkpoint()

    return llm


def do_inference_idefics(model, processor, prompts, max_new_tokens=50):
    # print('prompt:', prompts)
    # print('model:', model)
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        eos_token_id=[eos_token_id],
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        early_stopping=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(generated_text)
    answer = generated_text.split("Answer: ")[-1].strip()
    return answer


def handle_image(image_bytes, user_message):
    image_base64 = convert_bytes_to_base64(image_bytes)
    if os.path.basename(model_image_path).split('.')[-1] == 'gguf':
        output = multimodal.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant who perfectly describes images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": user_message}
                    ]
                }
            ]
        )
        print('ket qua cua anh: ', output)
        answer = output["choices"][0]["message"]["content"]
    else:
        im_PIL = base64_to_image(image_base64.split(',')[1])
        prompts = [
            im_PIL,
            f"Question: {user_message} Answer:",
        ]
        answer = do_inference_idefics(model=multimodal[0], processor=multimodal[1], prompts=prompts, max_new_tokens=500)

    # print('ket qua cua anh: ', output)
    return answer


# multimodal = load_llava()

if __name__ == "__main__":
    multimodal = load_checkpoint()
    image = Image.open(r"D:\QuanAI\dataset\PCBA\PCBA_New_name\Train\Paomain1_BOT_NG\Paomain1_BOT_NG_2.jpg")

    prompts = [
        image,
        "Question: Please analyze the image in detail? Answer:",
    ]
    answer = do_inference_idefics(model=multimodal[0], processor=multimodal[1], prompts=prompts, max_new_tokens=500)
    print(answer)
