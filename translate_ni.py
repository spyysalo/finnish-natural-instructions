#!/usr/bin/env python3

import sys
import re
import json

import torch

from copy import deepcopy
from argparse import ArgumentParser
from logging import warning

from sentence_splitter import SentenceSplitter
from transformers import pipeline


SPLIT_LINES_RE = re.compile(r'(\n\s*)')

LANG_CODE_MAP = {
    "ar": "Arabic",
    "as": "Assamese",
    "bn": "Bengali",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan",
    "km": "Central Khmer",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "tl": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "gu": "Gujarati",
    "ht": "Haitian",
    "he": "Hebrew",
    "hi": "Hindi",
    "ig": "Igbo",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "lo": "Lao",
    "ms": "Malay",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Oriya",
    "pa": "Panjabi",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "so": "Somali",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "yo": "Yoruba",
    "za": "Zhuang",
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--from_lang', default='en')
    ap.add_argument('--to_lang', default='fi')
    ap.add_argument('file', nargs='+')
    return ap


def load_task_data(fn, args):
    with open(fn) as f:
        data = json.load(f)
    return data


def filter_tasks(tasks, args):
    in_lang = LANG_CODE_MAP[args.from_lang]

    filtered = []
    for task in tasks:
        if task['Input_language'] == [in_lang]:
            filtered.append(task)

    print(f'filtered {len(tasks)} tasks to {len(filtered)}', file=sys.stderr)
    return filtered


def translate_text(text, pipe, args):
    splitter = SentenceSplitter(language=args.from_lang)

    try:
        lines_and_space = SPLIT_LINES_RE.split(text)
    except:
        warning(text)
        raise

    sentences_and_space = []
    for line in lines_and_space:
        if line.isspace() or not line:
            sentences_and_space.append(line)
        else:
            # NOTE: this loses the specifics of space between sentences
            sentences = splitter.split(line)
            space = [' '] * len(sentences)
            both = [s for ss in zip(sentences, space) for s in ss][:-1]
            sentences_and_space.extend(both)

    translated_and_space = []
    for sent in sentences_and_space:
        if sent.isspace() or not sent:
            translated_and_space.append(sent)
        else:
            translation = pipe(sent)[0]['translation_text']
            translated_and_space.append(translation)

    return ''.join(translated_and_space)


def translate(text_or_list, pipe, args):
    if isinstance(text_or_list, str):
        return translate_text(text_or_list, pipe, args)
    elif isinstance(text_or_list, list):
        return [translate_text(t, pipe, args) for t in text_or_list]


def translate_task(task, pipe, args):
    # Translate the parts of task data where MT is relevant. Roughly
    # preserve order to facilitate diff etc.
    
    mt = lambda t: translate(t, pipe, args)

    translated = {}

    # copy initial metadata
    for i in ('Contributors', 'Source', 'URL', 'Categories', 'Reasoning'):
        translated[i] = deepcopy(task[i])

    # translate definition
    for i in ('Definition',):
        translated[i] = mt(task[i])
        translated[f'original {i}'] = deepcopy(task[i])

    # map languages
    for i in ('Input_language', 'Output_language', 'Instruction_language'):
        translated[i] = LANG_CODE_MAP[args.to_lang]
        translated[f'original {i}'] = deepcopy(task[i])

    # copy domains
    for i in ('Domains',):
        translated[i] = deepcopy(task[i])
        
    # process examples and instances
    for i in ('Positive Examples', 'Negative Examples', 'Instances'):
        translated[i] = deepcopy(task[i])
        for e in translated[i]:
            for j in ('input', 'output'):
                e[f'original {j}'] = deepcopy(e[j])
                e[j] = mt(e[j])
                
    # copy license
    for i in ('Instance License',):
        translated[i] = deepcopy(task[i])
        
    return translated


def output_tasks(tasks, argv):
    for task in tasks:
        print(json.dumps(task, indent=4, ensure_ascii=False))


def main(argv):
    args = argparser().parse_args(argv[1:])

    if torch.cuda.is_available():
        device = 0
    else:
        device = None
        warning('cuda not available')
        
    model = f'Helsinki-NLP/opus-mt-{args.from_lang}-{args.to_lang}'
    pipe = pipeline('translation', model, device=device)
    
    tasks = []
    for fn in args.file:
        tasks.append(load_task_data(fn, args))

    tasks = filter_tasks(tasks, args)

    translated = [translate_task(t, pipe, args) for t in tasks]

    output_tasks(translated, args)
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))
