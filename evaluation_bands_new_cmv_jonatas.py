import re
from os.path import isfile, join
from os import listdir, walk
from datasets import Dataset, load_dataset, load_metric, concatenate_datasets
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
import torch 
import random
import pandas as pd
import csv
import warnings
from jiwer import cer
import jiwer

print("#### NEW CMMV")

def create_new_cmmv(path_to_csv, audio_folder):
    dict = {"path": [], "sentence": []}
    with open(path_to_csv, 'r') as f:

        read_tsv = csv.reader(f, delimiter="\t")
        read_tsv = [row for row in read_tsv]
        read_tsv = read_tsv[1:]
        for row in read_tsv:
            dict["path"].append(f"{audio_folder}{row[1]}")
            dict["sentence"].append(row[2])
    return Dataset.from_dict(dict)

print("#### Loading dataset")

print("####Commonvoice new")

fold = "../cv_new/cv-corpus-7.0-2021-07-21/it/"


common_voice_train = create_new_cmmv(f"{fold}train2.csv", f"{fold}clips2/")
common_voice_val = create_new_cmmv(f"{fold}dev2.csv", f"{fold}clips2/")  

print("####Merge datasets")

test_dataset = concatenate_datasets([common_voice_train, common_voice_val]).shuffle(seed=1234)


LANG_ID = "it"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
DEVICE = "cuda"

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

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

    wer_computed = wer.compute(predictions=[prediction.upper()], references=[batch["sentence"].upper()]) * 100
    cer_computed =jiwer.cer([prediction.upper()],[batch["sentence"].upper()]) * 100
    total_wer += wer_computed
    total_cer += cer_computed
    
    '''
    wer_computed = wer.compute(predictions=[prediction[0]], references=[batch["sentence"]]) * 100
    total_wer += wer_computed
    cer_computed= jiwer.cer(prediction, batch["sentence"])*100
    total_cer += cer_computed
    '''

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

with open(f"evaluation_bands_newcmv_total_jonatas.txt", "w") as f:
    f.write("New Common Voice Dataset Evaluation with jonatasgrosman/wav2vec2-large-xlsr-53-italian on train and valiadtion set \n")
    for key in sorted(ranges.keys()):
        mean_wer = ranges[key][1] / ranges[key][0]
        mean_cer = ranges[key][2] /ranges[key][0]
        f.write(f"[{int(key)*bands_len},{int(key)*bands_len+bands_len}) -> Count: {ranges[key][0]}, Wer: {mean_wer}, Cer: {mean_cer}\n")
    f.write(f"Total WER: {total_wer}  Total CER: {total_cer}\n")
    f.write(f"Dataset Len: {len(test_dataset)}\n")
