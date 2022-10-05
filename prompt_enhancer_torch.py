import json
import numpy as np
import os
import pandas as pd
import random
import re
from selenium import webdriver
import sys
import time
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy


class DescriptionDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length, is_mj_trunc = None):
        if is_mj_trunc is None:
            is_mj_trunc = [False]*len(txt_list)
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for idx, txt in enumerate(txt_list):
            removed_truncation_symbol = False
            # mj and len 203 or 252 and ending in ...
            if is_mj_trunc[idx] and (len(txt)>=203) and txt.endswith('...'):
                txt = txt[:-3]
                removed_truncation_symbol = True
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True, max_length=max_length, padding='max_length')
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            if removed_truncation_symbol:  # remove attention from end of text if we truncated
                eot_idx = len(encodings_dict['attention_mask']) - 1 - encodings_dict['attention_mask'][::-1].index(1)
                encodings_dict['attention_mask'][eot_idx] = 0
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


"""
Extract and save prompts from html:
Find every prompt in the raw html text
Load existing prompts if any from text file
Remove duplicates, sort
Write each one to a line of a text file
"""
def add_prompts_from_html(html_in, lexica_prompts_path):
    # prompts are in this format: "prompt":"[PROMPT IS HERE]"
    # find all instances of this format
    prompts = re.findall(r'\"prompt\":\"(.*?)\"', html_in)
    prompts = [prompt.replace('\\u0026','&').replace('\\u003e','>').replace('\\\\','\\') for prompt in prompts]
    num_prompts = len(prompts)

    # Load existing prompts if they exist
    if os.path.exists(lexica_prompts_path):
        with open(lexica_prompts_path, 'r', encoding='utf8') as f:
            old_prompts = f.read().splitlines()
    else:
        old_prompts = []
    num_old = len(old_prompts)

    # remove duplicates
    prompts = list(set(prompts + old_prompts))
    num_total = len(prompts)
    num_new = num_total - num_old
    if num_prompts == 0:
        fraction_new = 0
    else:
        fraction_new = num_new / num_prompts

    # sort
    prompts = sorted(prompts)

    # write to file
    with open(lexica_prompts_path, 'w', encoding='utf8') as f:
        for prompt in prompts:
            f.write(prompt + '\n')

    return num_new, num_prompts


"""
use the trained model to complete prompts
"""
def complete_prompt(prompt, num_return_sequences=5, tokenizer=None, model=None):
    tokenizer, model = load_tok_and_model(tokenizer=tokenizer, model=model)

    input_text = '<|startoftext|>' + prompt
    start_time = time.time()
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to('cuda')  # aka. generated
    outputs = model.generate(input_ids, attention_mask=inputs['attention_mask'].to('cuda'), do_sample=True,
                            max_length=500, temperature=0.8, use_cache=True, top_p=0.9,
                            num_return_sequences=num_return_sequences, pad_token_id=tokenizer.eos_token_id)
    print('Generate Time => ', time.time() - start_time, 'sec')

    for i, output in enumerate(outputs):
        print('{}: {}'.format(i, tokenizer.decode(output, skip_special_tokens=True)))

    return


"""
use the trained model to find a list of possible next portions to prompt
"""
def extend_prompt(prompt, num_return_suffixes=20, tokenizer=None, model=None, verbose = False):
    tokenizer, model = load_tok_and_model(tokenizer=tokenizer, model=model)
    num_return_sequences = 1000

    input_text = '<|startoftext|>' + prompt
    start_time = time.time()
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to('cuda')  # aka. generated
    outputs = model.generate(input_ids, attention_mask=inputs['attention_mask'].to('cuda'), do_sample=True,
                            max_new_tokens=30, temperature=0.8, use_cache=True, top_p=0.9,
                            num_return_sequences=num_return_sequences, pad_token_id=tokenizer.eos_token_id)
    print('Generate Time => ', time.time() - start_time, 'sec')

    eot_count = 0
    suffixes = []
    # decode and postproc
    for i, output in enumerate(outputs):
        str_dec = tokenizer.decode(output, skip_special_tokens=True)
        # suffix only, remove trailing whitespace
        suffix = str_dec[len(prompt):].rstrip()
        if suffix == '':
            eot_count += 1
        # ignore eot and single character suffixes
        if not (suffix=='' or len(suffix)==1):
            # truncate after next comma
            first_comma_idx = suffix[1:].find(',') + 1
            if first_comma_idx > 0:
                suffix_trunc = suffix[:first_comma_idx]
            else:
                suffix_trunc = suffix
            if suffix_trunc not in prompt:
                suffixes.append(suffix_trunc)

    # remove duplicates and sort by frequency
    suffixes = sorted(set(suffixes), key = lambda ele: -suffixes.count(ele))

    # top k
    if len(suffixes) >= num_return_suffixes:
        suffixes = suffixes[:num_return_suffixes]

    if verbose:
        for i, suffix in enumerate(suffixes):
            str_out = prompt + suffix
            print('{}: {}'.format((i+1), str_out))

    return suffixes, eot_count/num_return_sequences


def load_tok_and_model(tokenizer=None, model=None):
    if tokenizer is not None and model is not None:
        return tokenizer, model

    torch.manual_seed(42)
    model_str, lexica_prompts_path, midjourney_prompts_path, fine_tuned_model, fine_tuned_tokenizer = read_ini()

    fine_tuned_model_exists = os.path.exists(fine_tuned_model + '/pytorch_model.bin')
    fine_tuned_tokenizer_exists = os.path.exists(fine_tuned_tokenizer + '/tokenizer.json')

    if tokenizer is None and not fine_tuned_tokenizer_exists:
        print('Tokenizer not found at', fine_tuned_tokenizer)
        return 1
    if model is None and not fine_tuned_model_exists:
        print('Model not found at', fine_tuned_model)
        return 1

    if tokenizer is None:
        print('Loading tokenizer gpt-neo-' + model_str + '(fine-tuned)...', end='')
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_tokenizer)
        print('done')

    if model is None:
        start_time = time.time()
        print('Loading model gpt-neo-' + model_str + '(fine-tuned)...', end='')
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_model, use_cache=True).to('cuda')
        print('done')
        print('Load Time => ', time.time() - start_time, 'sec')

    return tokenizer, model


def read_ini():
    f = open('config.json')
    data = json.load(f)

    model_str = data['gpt_neo_version']  # 125M / 1.3B / 2.7B
    model_folder = data['model_folder']
    lexica_prompts_path = data['lexica_prompts']
    midjourney_prompts_path = data['midjourney_prompts']

    fine_tuned_model = model_folder + '/GPT-Neo-' + model_str + '_prompt_enh'
    fine_tuned_tokenizer = model_folder + '/GPT-Neo-' + model_str + '_prompt_enh'

    return model_str, lexica_prompts_path, midjourney_prompts_path, fine_tuned_model, fine_tuned_tokenizer


"""
read in stable diffusion prompts from text file, 1 per line
"""
def read_prompts(prompts_files, lexica_prompts_path, midjourney_prompts_path):
    all_descriptions = []
    is_mj_trunc = []
    for prompts_file in prompts_files:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            descriptions = f.read().lower().splitlines()
        if prompts_file == lexica_prompts_path:
            for idx, description in enumerate(descriptions):
                descriptions[idx].replace('\\n', '\n')  # need to handle newlines within a prompt after splitlines
        if prompts_file == midjourney_prompts_path:
            is_mj_trunc += [True]*len(descriptions)
        else:
            is_mj_trunc += [False]*len(descriptions)
        all_descriptions += descriptions

    return all_descriptions, is_mj_trunc


"""
Open https://lexica.art/?
Scrape the page source
Extract and save prompts
Wait 10 sec and repeat (Repeat until less than 50% of the prompts are new prompts)
"""
def scrape_lexica(frequency=10):
    model_str, lexica_prompts_path, midjourney_prompts_path, fine_tuned_model, fine_tuned_tokenizer = read_ini()

    fraction_new = 1
    while fraction_new >= -1:
        # Open https://lexica.art/?
        url = 'https://lexica.art/'
        driver = webdriver.Chrome('./chromedriver.exe')
        driver.get(url)
        time.sleep(frequency)

        # scrape the page source
        html_in = driver.page_source
        driver.quit()

        # extract and save prompts
        num_new, num_prompts = add_prompts_from_html(html_in, lexica_prompts_path)
        if num_prompts == 0:
            fraction_new = 0
        else:
            fraction_new = num_new / num_prompts
        print(num_new, '/', num_prompts, 'new prompts')

    return


def scrape_mj_favs(mj_favs_scrape_txt_path):
    model_str, lexica_prompts_path, midjourney_prompts_path, fine_tuned_model, fine_tuned_tokenizer = read_ini()

    with open(mj_favs_scrape_txt_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    prompts = re.findall(r'\*\*(.*?)\*\*', text)

    # remove prompts that end in ...
    prompts = [prompt for prompt in prompts if not prompt.endswith('...')]

    num_prompts = len(prompts)

    # Load existing prompts if they exist
    if os.path.exists(midjourney_prompts_path):
        with open(midjourney_prompts_path, 'r', encoding='utf8') as f:
            old_prompts = f.read().splitlines()
    else:
        old_prompts = []
    num_old = len(old_prompts)

    # remove duplicates
    prompts = list(set(prompts + old_prompts))
    num_total = len(prompts)
    num_new = num_total - num_old
    if num_prompts == 0:
        fraction_new = 0
    else:
        fraction_new = num_new / num_prompts

    # sort
    prompts = sorted(prompts)

    # write to file
    with open(midjourney_prompts_path, 'w', encoding='utf8') as f:
        for prompt in prompts:
            f.write(prompt + '\n')

    return num_new, num_prompts


"""
train a neural network to predict likely prompt completions
"""
def train_autocomplete():
    model_str, lexica_prompts_path, midjourney_prompts_path, fine_tuned_model, fine_tuned_tokenizer = read_ini()

    torch.manual_seed(42)
    fine_tuned_model_exists = os.path.exists(fine_tuned_model + '/pytorch_model.bin')
    fine_tuned_tokenizer_exists = os.path.exists(fine_tuned_tokenizer + '/tokenizer.json')

    print('Loading tokenizer gpt-neo-' + model_str, end='')
    if not fine_tuned_tokenizer_exists:
        print('(pretrained)...', end='')
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-' + model_str, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')  # unk_token?
        print('done')
        print('Saving tokenizer...', end='')
        tokenizer.save_pretrained(fine_tuned_tokenizer)
        print('done')
    else:
        print('(fine-tuned)...', end='')
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_tokenizer)
        print('done')

    start_time = time.time()
    print('Loading model gpt-neo-' + model_str, end='')
    if not fine_tuned_model_exists:
        print('(pretrained)...', end='')
        if model_str == '125M':
            use_cache = True
        else: # 1.3B
            use_cache = False
        model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-' + model_str, use_cache=use_cache).to('cuda')
        model.resize_token_embeddings(len(tokenizer))  # this won't work without retraining
    else:
        print('(fine-tuned)...', end='')
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_model, use_cache=True).to('cuda')
    print('done')
    print('Load Time => ', time.time() - start_time, 'sec')

    if not fine_tuned_model_exists:
        print('Loading prompts...', end='')
        descriptions, is_mj_trunc = read_prompts([lexica_prompts_path, midjourney_prompts_path],lexica_prompts_path, midjourney_prompts_path)
        print('done.\nMax length:', end='')
        max_length = max([len(tokenizer.encode(description)) for description in descriptions])
        print(' {}'.format(max_length))

        print('Fine tuning model:')
        dataset = DescriptionDataset(descriptions, tokenizer, max_length=max_length, is_mj_trunc=is_mj_trunc)
        train_size = int(0.95 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        if model_str=='125M':
            training_args = TrainingArguments(output_dir='./results', num_train_epochs=1, logging_steps=1000,
                                          save_strategy=IntervalStrategy.NO,
                                          per_device_train_batch_size=16, per_device_eval_batch_size=1,
                                          warmup_steps=100, weight_decay=0.01, logging_dir='./logs')
        else: # 1.3B
            training_args = TrainingArguments(output_dir='./results', num_train_epochs=1, logging_steps=1000,
                                              save_strategy=IntervalStrategy.NO,
                                              per_device_train_batch_size=4, per_device_eval_batch_size=1,
                                              warmup_steps=100, weight_decay=0.01, logging_dir='./logs',
                                              gradient_accumulation_steps=4, gradient_checkpointing=True)
        Trainer(model=model, args=training_args, train_dataset=train_dataset,
                eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                      'attention_mask': torch.stack([f[1] for f in data]),
                                                                      'labels': torch.stack([f[0] for f in data])}).train()
        print('Saving model...', end='')
        model.save_pretrained(fine_tuned_model)
        print('done')
        if model_str == '1.3B':  # if we don't reload it complains about use_cache=True and requires_grad!=True and crashes
            start_time = time.time()
            print('Reloading model gpt-neo-' + model_str, end='')
            model = AutoModelForCausalLM.from_pretrained(fine_tuned_model, use_cache=True).to('cuda')
            print('done')
            print('Load Time => ', time.time() - start_time, 'sec')

    print('Generating sample results...')
    input_text = '<|startoftext|>Spiderman using a giant ruler to measure a web,'
    start_time = time.time()
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to('cuda')  # aka. generated
    sample_outputs = model.generate(input_ids, attention_mask=inputs['attention_mask'].to('cuda'), do_sample=True,
                            max_length=400, temperature=0.8, use_cache=True, top_p=0.9, num_return_sequences=2,
                            pad_token_id=tokenizer.eos_token_id)
    print('Generate Time => ', time.time() - start_time, 'sec')

    for i, sample_output in enumerate(sample_outputs):
        print('{}: {}'.format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

    return


import tkinter
if __name__ == '__main__':
    # scrape_mj_favs('./Midjourney - Image Galleries - favorites [938154212238430341] (after 2022-01-01).txt')
    # scrape_lexica(frequency=20)  # let this run for while to get quality prompts
    # train_autocomplete()  #  fine-tune the model for simple prompt generation
    # complete_prompt('Spiderman using a giant ruler to measure a web,')  # finish a prompt
    # extend_prompt('Spiderman using a giant ruler to measure a web', verbose=True)  # finish a prompt

    # get prompt from text box
    tokenizer, model = load_tok_and_model()
    root = tkinter.Tk()
    root.title('Prompt Enhancer')
    root.geometry('720x200')
    root.resizable(True, False)

    prompt_label = tkinter.Label(root, text='Prompt:')
    prompt_label.place(x=10, y=10)

    prompt_entry = tkinter.Entry(root, width=115)
    prompt_entry.place(x=10, y=30)

    global dropdown
    dropdown = tkinter.Label(root, text='NOT VISIBLE')

    def get_prompt():
        global dropdown
        prompt = prompt_entry.get()
        print('Prompt:', prompt)
        suffixes, eot_chance = extend_prompt(prompt, tokenizer=tokenizer, model=model)
        eot_pct = int(100 * eot_chance)

        # create dropdown list of suffixes
        def set_text(suffix):
            global dropdown
            prompt_entry.delete(0, tkinter.END)
            prompt_entry.insert(0, prompt + suffix)
            get_prompt()
        if len(suffixes) > 0:
            var = tkinter.StringVar()
            var.set('<select suffix> - ~' + str(eot_pct) + '% End of Text')
            dropdown.destroy()
            dropdown = tkinter.OptionMenu(root, var, *suffixes, command=set_text)
            dropdown.place(x=10, y=60)
        else:
            dropdown.destroy()
            dropdown = tkinter.Label(root, text='No likely continuations...')
            dropdown.place(x=10, y=60)

        return

    button = tkinter.Button(root, text='Resubmit Prompt', command=get_prompt)
    button.place(x=10, y=100)

    def press_return(event):
        get_prompt()

    root.bind('<Return>', press_return)

    def close_window():
        root.destroy()

    button_close = tkinter.Button(root, text='Close', command=close_window)
    button_close.place(x=10, y=150)

    def copy_prompt():
        root.clipboard_clear()
        root.clipboard_append(prompt_entry.get())

    button_copy = tkinter.Button(root, text='Copy Prompt\nto Clipboard', command=copy_prompt)
    button_copy.place(x=70, y=150)

    root.mainloop()
