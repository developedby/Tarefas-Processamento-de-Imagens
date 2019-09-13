"""Tarefa 3 de processamento digital de imagens
Descrição: Implementa efeito bloom
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
INPUT_IMAGE = "GT2.BMP"
MEAN_FILTER_AMOUNT = 5
ALPHA = 0.3
BETTA = 0.7

if __name__ == '__main__':
    # Abre a imagem
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}", cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao abrir a imagem")
        exit()
    # Normaliza
    img = img.astype(float) / 255

    # Ingênuo
    initial_time = time.time()
    out_img = bloom(img, MEAN_FILTER_AMOUNT, ALPHA, BETTA)
    total_time = time.time() - initial_time
    print(f"Tempo: {total_time}")
    out_img = float_to_uint8(out_img)
    cv2.imwrite(f"{IMAGE_FOLDER}/01 - bloor.bmp", out_img)
