from pathlib import Path
from collections.abc import Callable
from typing import Any
from PIL import Image
from tqdm.auto import tqdm

import numpy as np
import torch
from torchvision.transforms import v2
from albumentations.core.transforms_interface import BasicTransform


class ColorShiftTorch(v2.Transform):
    def __init__(self, min_val: float = .5, p: float = .5):
        super().__init__()
        self.min_val = min_val
        self.p = p

    def _get_params(self, flat_inputs) -> dict[str, Any]:  # make_params
        apply_transform = (torch.rand(size=(1,)) < self.p).item()
        params = dict(apply_transform=apply_transform)
        return params

    def _transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor: # transform
        if params['apply_transform']:
            return self._color_shift(inpt, self.min_val)
        else:
            return inpt

    @staticmethod
    def _color_shift(img: torch.Tensor, min_val: float = .5) -> torch.Tensor:
        shift = min_val +  torch.rand(3, 1, 1) * (1 - min_val) # [min_val, 1]
        img = (255 - (255 - img) * shift).clip(0, 255).to(dtype=torch.uint8)
        return img


class ColorShiftAlb(BasicTransform):
    def __init__(
        self, 
        min_val: float = .5, 
        always_apply: bool | None = None, 
        p: float = .5
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.min_val = min_val

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self._color_shift(img, self.min_val)

    def get_transform_init_args_names(self):
        return("min_val",)

    @property
    def targets(self):
        return {"image": self.apply}

    @staticmethod
    def _color_shift(img: np.ndarray, min_val: float = .5) -> np.ndarray:
        shift = min_val +  np.random.rand(1, 1, 3) * (1 - min_val) # [min_val, 1]
        img = (255 - (255 - img) * shift).clip(0, 255).astype(np.uint8)
        return img


def edit_images(
    input_dir: Path, 
    output_dir: Path, 
    preprocess: Callable[..., Image.Image], 
    *args
) -> None:
    """Преобразование изображений в директории с использованием пользовательской функции.

    Args:
        input_dir: Путь к директории с исходными изображениями
        new_dataset: Путь к директории, куда будет сохранены измененные изображения.
        preprocess: Функция для преобразования изображений.
        *args: Дополнительные аргументы, которые передаются в 'preprocess'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_paths = input_dir.glob('*.png')
    images_paths = sorted(list(images_paths))
    
    for img_path in tqdm(images_paths):
        img = Image.open(img_path)
        # Обработка переданной функцией
        img = preprocess(img, *args)
        #  Сохранение изображения
        img.save(output_dir / img_path.name)