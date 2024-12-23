import json
import re
import random
from pathlib import Path
from io import BytesIO
from collections.abc import Callable
from typing import Any
from termcolor import colored

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import Levenshtein

plt.rcParams.update({
    'text.usetex': True,
    # 'mathtext.fontset': 'cm'
})


class Ansi:
    green = '\033[32m'
    red = '\033[31m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'


def write_json(path: str | Path, obj: Any) -> None:
    """Запись объекта в *.json."""
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(obj, file, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    """Чтение *.json файла."""
    with open(path, 'r', encoding='utf-8') as file:
        obj = json.load(file)
    return obj

#------------------
# Texts
#------------------
def highlight_diff(gt: str, pred: str) -> str:
    """Выделение различий в двух строках.

    Args:
        gt: Ground truth text.
        pred: Prediction text.

    Returns:
        Срока, в котором с помощью ansi-кодов будут отмечены отличия в текстах:
            - красным цветом - добавленные или замененные символы
            - светло серым цветом - удаленные символы
    """
    opcodes = Levenshtein.opcodes(gt, pred)
    result = []
    for tag, i1, i2, j1, j2 in opcodes:
        gt_fragment = gt[i1:i2]
        pred_fragment = pred[j1:j2]

        if tag == 'equal':
            result.append(pred_fragment)

        elif tag == 'delete':
            if re.fullmatch('\s+', gt_fragment):
                result.append(colored('_' * (i2 - i1), 'light_grey', attrs=['bold']))
            else:
                result.append(colored(gt_fragment, 'light_grey'))

        elif tag in ('replace', 'insert'):
            if re.fullmatch('\s+', pred_fragment):
                result.append(colored('_' * (j2 - j1), 'red', attrs=['bold']))
            else:
                result.append(colored(pred_fragment, 'red'))
                
    return ''.join(result)


#------------------
# Torch
#------------------
def torch_device() -> str:
    """Вовзращает строку 'cuda' при наличии оборудования, иначе 'cpu'."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"torch use: {colored(device, 'green', attrs=['bold'])}", 
          f"({torch.cuda.get_device_name()})"
          if torch.cuda.is_available() else "")
    return device


class DeNormalize(v2.Normalize):
    """Денормализации изображений."""
    def __init__(self,mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


#------------------
# Latex
#------------------
def process_raw_latex(sequence: str) -> str:
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        sequence: Input string.
    """
    sequence = re.sub(r'\\label\s*\{.*?}\s*', '', sequence)
    # sequence = re.sub(r'\$', '', sequence)
    # sequence = re.sub(r'\\mbox', r'\\mathrm', sequence)

    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, sequence)]
    sequence = re.sub(text_reg, lambda match: str(names.pop(0)), sequence)

    def clean_whitespace(seq: str) -> str:
        seq = re.sub(fr'(?!\\ )({noletter})\s+?({noletter})', r'\1\2', seq)
        seq = re.sub(fr'(?!\\ )({noletter})\s+?({letter})', r'\1\2', seq)
        seq = re.sub(fr'({letter})\s+?({noletter})', r'\1\2', seq)
        return seq

    while True:
        newseq = clean_whitespace(sequence)
        if newseq == sequence:
            break
        sequence = newseq

    return sequence


def latex2pil(sequence: str, fontsize: int = 25) -> Image.Image:
    """Перевод LaTeX в изображение PIL."""

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(.1, .1))
    ax.text(0.5, 0.5, f'${sequence}$', fontsize=fontsize) 
    ax.axis(False)

    # Сохраняем изображение в буфер
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Перемещаем указатель в начало буфера
    buffer.seek(0)
    
    # Создаем объект PIL.Image из буфера
    img = Image.open(buffer).convert('RGB')
    buffer.close()
    
    return img


#------------------
# Auxiliary functions for the project
#------------------
def _plot_img_and_formula(img: Image.Image, formula: str, figsize=(12, .8)) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis(False)
    plt.show()
    print(f"{colored('img size', attrs=['bold'])}: {img.size}")
    print(f"{colored('formula', attrs=['bold'])}: {formula}")
    print('=' * 70)


def draw_random_images(img_dir: Path, annotations: Path, nums: int = 5) -> None:
    annotations = read_json(annotations)
    img_names = random.choices(list(annotations.keys()), k=nums)

    for img_name in img_names:
        img = Image.open(img_dir / img_name)
        _plot_img_and_formula(img, annotations[img_name])


def draw_random_images_torchset(dataset: Dataset, nums: int = 5) -> None:
    denorm = DeNormalize(
        mean=dataset.processor.image_processor.image_mean, 
        std=dataset.processor.image_processor.image_std
    )
    indices = random.choices(range(len(dataset)), k=nums)

    for idx in indices:
        img, target = dataset[idx]
        img = denorm(img.squeeze())
        img = v2.functional.to_pil_image(img)
        _plot_img_and_formula(img, target['formula'], figsize=(12, 1.5))