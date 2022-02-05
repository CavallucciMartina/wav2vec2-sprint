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

main_dir = "../mls_italian_opus"
test_dir = join(main_dir, 'test')
train_dir = join(main_dir, 'train')
file_transcripts = 'transcripts.txt'
LANG_ID = "it"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
DEVICE = "cuda"

def create_hug_dataset(split_directory):
    
    list_opus = []
    labels_dict = {}
    for (dirpath, dirnames, filenames) in walk(split_directory):
        list_opus += [join(dirpath, file) for file in filenames if file.endswith(".opus")]

    with open(join(split_directory, file_transcripts), 'r') as f: 
        content = f.read()
        sentences = content.split(sep="\n")

    for sent in sentences:
        if(sent != ''):
            sent = re.sub(' +', ' ', sent)
            sent = sent.split("\t", maxsplit=1)
            labels_dict[sent[0]] = sent[1]

    audio_dict = {opus.split("/")[-1].split(".")[0]: opus for opus in list_opus}

    #print("#### Removing special characters from labels mlls")

    #labels_dict = {k: remove_special_characters_mlls(v) for k, v in labels_dict.items()}
    dict_dataset = {'path': [], 'sentence': []}

    for k, v in audio_dict.items():
        dict_dataset['path'].append(v)
        dict_dataset['sentence'].append(labels_dict[k])

    #tot_len = len(dict_dataset["path"])

    return Dataset.from_dict(dict_dataset)

print("#### Loading dataset (train + test")
mls_train = create_hug_dataset(train_dir)
mls_test = create_hug_dataset(test_dir)


print("####Merge datasets")

merged_dataset = concatenate_datasets([mls_train, mls_test]).shuffle(seed=1234)

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

'''
# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).upper()
    return batch

merged_dataset = merged_dataset.map(speech_file_to_array_fn)
'''

ranges = {} # contains (count, tot_wer,total_cer)

bands_len = 2 #2 second bands

total_wer=0
total_cer=0
for index, batch in enumerate(merged_dataset):
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


total_wer /= len(merged_dataset)
total_cer /= len(merged_dataset)

with open(f"evaluation_bands_mls_total_jonatas.txt", "w") as f:
    f.write("MLS Dataset Evaluation with jonatasgrosman/wav2vec2-large-xlsr-53-italian on train and test set \n")
    for key in sorted(ranges.keys()):
        mean_wer = ranges[key][1] / ranges[key][0]
        mean_cer = ranges[key][2] /ranges[key][0]
        f.write(f"[{int(key)*bands_len},{int(key)*bands_len+bands_len}) -> Count: {ranges[key][0]}, Wer: {mean_wer}, Cer: {mean_cer}\n")
    f.write(f"Total WER: {total_wer}  Total CER: {total_cer}\n")
    f.write(f"Dataset Len: {len(merged_dataset)}\n")