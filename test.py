import torch
# from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint = "HuggingFaceM4/tiny-random-idefics"
base_model_id = r"D:\QuanAI\Models\idefics_9b_instruct"
peft_model_id = r'D:/QuanAI/Models/idefics_9b_instruct_CBD/QuanAI-CBD-PCBA-Images/full'

# Here we skip some special modules that can't be quantized properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)

processor = AutoProcessor.from_pretrained(base_model_id, use_auth_token=True)
# Simply take-off the quantization_config arg if you want to load the original model
model = IdeficsForVisionText2Text.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

peft_model = PeftModel.from_pretrained(model
                                       , peft_model_id,
                                       is_trainable=False)


# model.load_adapter(r'D:\QuanAI\Models\idefics_9b_instruct_CBD\QuanAI-CBD-PCBA-Images\checkpoint-50')
# model.enable_adapters()

def do_inference(model, processor, prompts, max_new_tokens=50):
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
    print(generated_text)


url = r"D:\QuanAI\dataset\PCBA\PCBA_New_name\Train\Sanrepain_BOT_NG\Sanrepain_BOT_NG_2.jpg"
image = Image.open(url)
prompts = [
    image,
    "Question: Please analyze the image in detail? Answer:",
]

do_inference(peft_model, processor, prompts, max_new_tokens=500)
print('OK')
