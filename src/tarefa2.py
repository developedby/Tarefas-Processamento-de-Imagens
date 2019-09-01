"""Tarefa 2 de processamento digital de imagens
Descrição: Implementa filtro da média usando método ingênuo, separável mantendo
anterior e com imagem integral
Autores: Nicolas Abril e Álefe Felipe Gonçalves Dias
Professor: Bogdan Nassu
Engenharia de Computação UTFPR-CT
09/2019
"""
import time
import cv2
import numpy as np
from pdi import *

# Parâmetros
IMAGE_FOLDER = "../img"
INPUT_IMAGE = "flowers.bmp"
WINDOW_HEIGHT = 3
WINDOW_WIDTH = 3

if __name__ == '__main__':
    # Abre a imagem
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}", cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao abrir a imagem")
        exit()
    # Normaliza
    img = img.astype(float) / 255
    window = [WINDOW_HEIGHT, WINDOW_WIDTH]

    # Ingênuo
    initial_time = time.time()
    out_img = mean_filter_bad(img, window)
    total_time = time.time() - initial_time
    print(f"Tempo ingenuo: {total_time}")
    out_img = (out_img*255).astype(np.uint8)
    cv2.imwrite(f"{IMAGE_FOLDER}/01 - ingenuo.bmp", out_img)

    # Separavel
    initial_time = time.time()
    out_img = mean_filter_separable(img, window)
    total_time = time.time() - initial_time
    print(f"Tempo separavel: {total_time}")
    out_img = (out_img*255).astype(np.uint8)
    cv2.imwrite(f"{IMAGE_FOLDER}/02 - separavel.bmp", out_img)

    # Integral
    initial_time = time.time()
    out_img = mean_filter_integrated(img, window)
    total_time = time.time() - initial_time
    print(f"Tempo integral: {total_time}")
    out_img = (out_img*255).astype(np.uint8)
    cv2.imwrite(f"{IMAGE_FOLDER}/03 - integral.bmp", out_img)
