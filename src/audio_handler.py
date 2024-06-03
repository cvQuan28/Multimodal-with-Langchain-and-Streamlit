import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import io
from utils import load_config
import soundfile as sf
from pydub import AudioSegment
import time

config = load_config()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def convert_bytes_to_array(audio_bytes):
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes)
    # print(sample_rate)
    return audio


def transcribe_audio(audio_bytes):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = config["whisper_model"]
    # pipe = pipeline(
    #     task="automatic-speech-recognition",
    #     model=config["whisper_model"],
    #     chunk_length_s=30,
    #     device=device,
    # )
    if device == "cpu":
        # device = "cpu"
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=config["whisper_model"],
            chunk_length_s=30,
            device=device,
        )
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
    # # Chuyển đổi dữ liệu âm thanh từ dạng bytes thành đối tượng AudioSegment
    # audio_segment = AudioSegment.from_bytes(io.BytesIO(audio_bytes))
    # # Lưu dữ liệu âm thanh thành file âm thanh
    # audio_segment.export("output.mp3", format="mp3")

    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=1)["text"]

    return prediction


if __name__ == "__main__":
    audio_path = r"C:\Users\Admin\Downloads\828694b5e055ee832a3ccaefb5ab680c390f1d71d0deb51ad74696f0.wav"


    def load_audio_to_bytes(file_path):
        with open(file_path, 'rb') as file:
            audio_bytes = file.read()
        return audio_bytes


    audio_bytes = load_audio_to_bytes(audio_path)

    re = transcribe_audio(audio_bytes)
