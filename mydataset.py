import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import v2
from doctr import transforms as T

from utils import read_json, process_raw_latex


class NougatDataset(Dataset):
    def __init__(
        self,
        img_dir: Path | str,
        annotations: Path | str,
        processor,
        transforms=None,
        max_length: int = 800,
        **kwargs
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.max_length = max_length
        self.processor = processor

        annotations_raw = read_json(Path(annotations))
        self.annotations = {}

        for img_name in annotations_raw.keys():
            if not self.img_dir.joinpath(img_name).exists():
                raise FileNotFoundError(f"unable to locate {self.img_dir / img_name}")

        for img_name, formula in tqdm(annotations_raw.items()):
            # Удаление лишних пробелов из LaTeX
            formula = process_raw_latex(formula)
            tokenize = self.processor.tokenizer(formula, return_token_type_ids=False)
            if max_length >= len(tokenize.attention_mask):
                target = {
                    'input_ids': tokenize.input_ids,
                    'mask': tokenize.attention_mask,
                    'formula': formula,
                }
                self.annotations[img_name] = target

        self.img_names = list(self.annotations.keys())

        # # Масштабирование с сохранением пропорций и добавлением паддингов
        # img_size = tuple(processor.image_processor.size.values())
        # self.resize = v2.Compose([
        #     v2.ToImage(),
        #     T.Resize(img_size, preserve_aspect_ratio=True, symmetric_pad=True),
        #     v2.ToPILImage(),
        # ])
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx: int):
        img_name = self.img_names[idx]
        target = self.annotations[img_name]
        img = Image.open(self.img_dir / img_name).convert('RGB')

        if self.transforms is not None:
            img = np.array(img)
            img = self.transforms(image=img)['image']
            img = v2.functional.to_pil_image(img)
            
        # img = self.resize(img)
        img = self.processor.image_processor(img, return_tensors="pt").pixel_values
        
        return img, target

    def collate_fn(self, instances):
        images, input_ids, masks, formulas = [], [], [], []
        
        for img, target in instances:
            images.append(img)
            input_ids.append(torch.LongTensor(target['input_ids']))
            masks.append(torch.LongTensor(target['mask']))
            formulas.append(target['formula'])

        images = torch.cat(images, dim=0)
        # Паддинг меток и масок
        pad_token_id = self.processor.tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        
        return {
            'input_ids': input_ids,
            'masks': masks,
            'formulas': formulas,
            'pixel_values': images
        }