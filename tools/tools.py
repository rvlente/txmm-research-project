# -*- coding: utf-8 -*-
"""
tools.py
========

Tools for extracting text from books.

Usage:

$ python -m tools.py
>>> scans_to_images('gk-aeneis/all.pdf', 'gk-aeneis/source')
>>> extract_text('gk-aeneis/source/0.jpg', 'gk-aeneis/text/0.txt')
...
>>> combine_text('gk-aeneis/text', 'gk-aeneis/all.txt')
"""

from PIL import Image
import pdf2image
import pytesseract
from pathlib import Path


def extract_text(image_file, output_file, contrast_threshold=180, skip_validation=False):
    """
    Extract text from `image_file` and write it to `output_file`.

    If `skip_validation` is `True`, each extracted line is printed for validation.
    The user is presented with the following options:
    - Y (default): Write the line as is.
    - m: Write a marker before writing the line as is.
    - n: Skip the line.
    """

    image = Image.open(image_file)
    image = image.convert('L').point(lambda x: 255 if x > contrast_threshold else 0, mode='1')

    if not skip_validation:
        image.save('inspect.jpg')

    results = []
    text = pytesseract.image_to_string(image, lang='nld')

    for line in text.split('\n'):
        if not line.strip():
            continue

        if skip_validation:
            results.append(line)
        else:
            print(line)
            keep = input('(Y/m/n) ')
            if keep == 'Y' or keep == '':
                results.append(line)
            elif keep == 'm':
                results.append('###')
                results.append(line)

    with open(output_file, 'w') as f:
        for line in results:
            f.write(f'{line}\n')


def scans_to_images(input_file, output_dir):
    """
    From a single scan of all pages in pdf format (`input_file`), extract all page images to `output_dir`.
    Images are halved horizontally and saved as numbered .jpg images.
    """

    images = pdf2image.convert_from_path(input_file)

    pages = []
    for image in images:
        w, h = image.size
        pages.append(image.crop((0, 0, w / 2, h)))
        pages.append(image.crop((w / 2, 0, w, h)))

    for i, p in enumerate(pages):
        p.save(f'{output_dir}/{i}.jpg')


def combine_text(input_folder, output_file):
    """
    Sequentially combine numbered txt files from `input_folder` into `output_file`.
    """

    with open(output_file, 'w') as output:
        for file in sorted(Path(input_folder).iterdir(), key=lambda x: int(x.stem)):
            with open(file) as f:
                output.write(f.read())
                output.write('\n')
