"""Tarefa 5 de processamento digital de imagens
Descrição: Faz chroma key
Autores: Nicolas Abril e Álefe Felipe Gonçalves Dias
Professor: Bogdan Nassu
Engenharia de Computação UTFPR-CT
10/2019
"""
import time
import sys
import cv2
import numpy as np
import pdi

# Parâmetros
IMAGE_FOLDER = "../img/tarefa5"
INPUT_IMAGE = "0.BMP"
BG_IMAGE = 'bg.jpeg'
INVERT_RGB = False

if __name__ == '__main__':
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}")
    bg_img = cv2.imread(f"{IMAGE_FOLDER}/{BG_IMAGE}")
    if img is None or bg_img is None:
        print("Erro ao abrir a imagem")
        sys.exit()
    h, w, _ = img.shape
    if INVERT_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)
    bg_img = cv2.resize(bg_img, (w, h))

    start_time = time.time()

    merge, mask, dists = pdi.green_screen(img, bg_img, extra_ret=False)
    if INVERT_RGB:
        merge = cv2.cvtColor(merge, cv2.COLOR_BGR2RGB)

    total_time = time.time() - start_time
    print(f"Tempo total: {total_time}")

    cv2.imwrite(f"{INPUT_IMAGE} - Result.bmp", merge)

    if mask is not None:
        cv2.imwrite(f"{INPUT_IMAGE} - Mask.bmp", mask)
    if dists is not None:
        cv2.imwrite(f"{INPUT_IMAGE} - Dists.bmp", pdi.float_to_uint8(dists))
