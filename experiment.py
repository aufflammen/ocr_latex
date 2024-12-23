import random
from pathlib import Path
from collections.abc import Callable
from typing import Any
from termcolor import colored
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchmetrics.functional.text import char_error_rate

from utils import read_json, latex2pil, highlight_diff, process_raw_latex


def show_predicts(
    img_dir: Path, 
    annotations: Path, 
    nums: str,
    predict_from_pil: Callable,
    **kwargs
):
    """Получение предсказаний для нескольких случайных изображений.

    Args:
        img_dir: Директория с изображениями.
        annotations: Путь к файлу с аннотациями.
        nums: Количество выводимых примеров.
        predict_from_pil: Функция для получения предсказания модели по одному изображению.
        **kwargs: именованные аргументы для predict_from_pil.

    Returns:
        Для каждого изображения из датасета будет отображено:
            - исходное изображение
            - изображение, отрендеренное по предсказанной формуле
            - истинная формула
            - предсказанная формула
            - [если CER > 0, то отобразится строка, в которой будут отмечены различия в формулах:
                - красным цветом - добавленные или замененные символы
                - светло серым цветом - удаленные символы]
            - метрика CER, вычисленная для формул
            
    """
    annotations = read_json(annotations)
    img_names = random.choices(list(annotations.keys()), k=nums)

    gt_title = colored('Ground truth:', attrs=['bold'])
    pred_title = colored('Predict:     ', attrs=['bold'])
    
    for img_name in img_names:
        gt_img = Image.open(img_dir.joinpath(img_name))
        gt_formula = annotations[img_name]
        gt_formula = process_raw_latex(gt_formula)
        # gt_tokens = processor.tokenizer(gt_formula, return_token_type_ids=False).input_ids

        pred_formula = predict_from_pil(img=gt_img, **kwargs)
        pred_img = latex2pil(pred_formula)

        print(gt_title)
        plt.figure(figsize=(12, .8))
        plt.imshow(gt_img)
        plt.axis(False)
        plt.show()
        
        print(pred_title)
        plt.figure(figsize=(12, .8))
        plt.imshow(pred_img)
        plt.axis(False)
        plt.show()

        cer = char_error_rate(pred_formula, gt_formula)
        print(f"\n{gt_title} {gt_formula}")
        print(f"{pred_title} {pred_formula}")
        
        if cer > 0:
            print(f"\n{colored('Difference:  ', attrs=['bold'])} {highlight_diff(gt_formula, pred_formula)}")

        print(f"\n{colored('CER', attrs=['bold'])}: {cer:.4f}")

        print('=' * 70)