import torch
import librosa
import re
import warnings
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import cer
import jiwer
import torchaudio

LANG_ID = "it"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
DEVICE = "cuda"

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

test_dataset = load_dataset("common_voice", LANG_ID, split="test")

wer = load_metric("wer.py") # https://github.com/jonatasgrosman/wav2vec2-sprint/blob/main/wer.py
cer = load_metric("cer") #For CER import from jiwer because jonatas code generated error

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.to(DEVICE)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

ranges = {} # contains (count, tot_wer,total_cer)

bands_len = 2 #2 second bands

total_wer=0
total_cer=0
for index, batch in enumerate(test_dataset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(pred_ids)
    wer_computed = wer.compute(predictions=[prediction[0]], references=[batch["sentence"]]) * 100
    total_wer += wer_computed
    cer_computed= jiwer.cer(prediction, batch["sentence"])*100
    total_cer += cer_computed

    info = torchaudio.info(batch["path"])
    duration_sec = info.num_frames / info.sample_rate

    band = int(duration_sec / bands_len)

    if band not in ranges:
        ranges[band] = [1, wer_computed,cer_computed]
    else:
        ranges[band][0] += 1
        ranges[band][1] += wer_computed
        ranges[band][2] += cer_computed


total_wer /= len(test_dataset)
total_cer /= len(test_dataset)

with open(f"evaluation_bands_cmmv_jonatas.txt", "w") as f:
    f.write("Common Voice Evaluation with jonatasgrosman/wav2vec2-large-xlsr-53-italian on test set \n")
    for key in sorted(ranges.keys()):
        mean_wer = ranges[key][1] / ranges[key][0]
        mean_cer = ranges[key][2] /ranges[key][0]
        f.write(f"[{int(key)*bands_len},{int(key)*bands_len+bands_len}) -> Count: {ranges[key][0]}, Wer: {mean_wer}, Cer: {mean_cer}\n")
    f.write(f"Total WER: {total_wer}  Total CER: {total_cer}\n")
    f.write(f"Dataset Len: {len(test_dataset)}\n")
