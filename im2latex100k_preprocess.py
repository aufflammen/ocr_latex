from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

from utils import write_json, process_raw_latex, latex2pil


def read_txt(path: Path | str) -> None:
    with open(path, 'r', encoding='utf-8') as file:
        text = file.readlines()
    return text


def trim_image(img: Image.Image, padding: int = 10) -> Image.Image | None:
    """Обрезает изображение с отступом от найденной границы.
    
    Args:
        image: Входное изображение в RGBA.
        padding: Количество пикселей для отступа.
        
    Returns:
        PIL.Image: Обрезанное изображение с отступом.
    """
    alpha = img.split()[-1]
    bbox = alpha.getbbox()

    # Если границ нет (полностью пустое изображение), возвращаем None
    if not bbox:
        return None
        
    left, upper, right, lower = bbox
    # Добавляем отступы
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(img.width, right + padding)
    lower = min(img.height, lower + padding)
    # Обрезаем изображение с учетом отступов
    cropped_img = img.crop((left, upper, right, lower))
    return cropped_img



def rgba2rgb(img: Image.Image) -> Image.Image:
    """Конвертирует изображение из RGBA в RGB без размытия границ.

    Args:
        img: Входное изображение в формате RGBA.
        background_color: Цвет фона для прозрачных областей (по умолчанию белый).

    Returns:
        PIL.Image.Image: Изображение в формате RGB.
    """
    if img.mode != 'RGBA':
        raise ValueError("Изображение должно быть в формате RGBA.")
    
    # Создаем фон
    background_color = (255,) * 4
    background = Image.new("RGBA", img.size, background_color)
    # Накладываем изображение поверх фона
    rgb_image = Image.alpha_composite(background, img)
    return rgb_image.convert('RGB')


class Im2LatexPreprocess:
    """ 
    Конвертация датасета im2latex-100k (https://zenodo.org/records/56198#.V2px0jXT6eA)
    в более удобный формат.

    ---BEFORE:---
    dataset
    ├── formula_images
    ├── im2latex_formulas.lst
    ├── im2latex_test.lst
    ├── im2latex_train.lst
    └── im2latex_validate.lst

    ---AFTER:---
    dataset
    ├── test
    │   ├── images
    │   └── annotations.json
    ├── train
    │   ├── images
    │   └── annotations.json
    └── val
        ├── images
        └── annotations.json
    """
    
    def __init__(
        self, 
        input_dir: Path, 
        output_dir: Path, 
        num_workers: int | None = None
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.formulas = self._get_formulas()

    
    def _get_formulas(self):
        formulas = []
        
        with open(self.input_dir / 'im2latex_formulas.lst', 'rb') as file:
            for line in file:
                try:
                    formula = line.decode('utf-8').rstrip('\n')
                    formula = process_raw_latex(formula)
                    formulas.append(formula)
                except:
                    formulas.append(None)

        return formulas

    @staticmethod
    def _process_item(input_img_path, output_dir: Path, formula) -> tuple[str, str] | None:
        img = Image.open(input_img_path).convert('RGBA')
        img = trim_image(img, 10)
        if img is None:
            return None

        img_name = input_img_path.name
        img = rgba2rgb(img)
        img.save(output_dir / 'images' / img_name)
        return img_name, formula

    
    def convert_stage(
        self,
        input_annot: Path, 
        output_dir: Path,
    ) -> None:
        output_dir.joinpath('images').mkdir(parents=True, exist_ok=True)
        input_annot = read_txt(input_annot)
        input_annot = [s.strip().split()[:2] for s in input_annot]
        annotations = {}
    
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for idx, name in input_annot:
                idx = int(idx)
                img_name = f'{name}.png'
                formula = self.formulas[idx]
                if formula is not None:
                    input_img_path = self.input_dir / 'formula_images' / img_name
                    futures.append(executor.submit(self._process_item, input_img_path, output_dir, formula))
                    
            # Обработка результатов
            for future in tqdm(futures, desc=output_dir.name):
                result = future.result()
                if result is not None:
                    annotations[result[0]] = result[1]
    
        write_json(output_dir / 'annotations.json', annotations)

        
    def convert(self) -> None:
        
        self.convert_stage(self.input_dir / 'im2latex_train.lst', 
                           self.output_dir / 'train')
        
        self.convert_stage(self.input_dir / 'im2latex_validate.lst', 
                           self.output_dir / 'val')
        
        self.convert_stage(self.input_dir / 'im2latex_test.lst', 
                           self.output_dir / 'test')