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
from pdi import bloom, approx_gaussian_blur

# Parâmetros
IMAGE_FOLDER = "../img/tarefa3"
INPUT_IMAGE = "Wind Waker GC.bmp"
THRESHOLD = 0.5
NUM_ITER_MEAN = 3
BLOOM_WEIGHT = 0.03
NUM_BLURS = 5
INIT_SIGMA = 3

if __name__ == '__main__':
    # Abre a imagem
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}", cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao abrir a imagem")
        exit()

    # Gaussiana
    blur_func = lambda img, sigma: cv2.GaussianBlur(img, (0, 0), sigma)
    initial_time = time.time()
    out_img = bloom(img, THRESHOLD, NUM_BLURS, INIT_SIGMA, BLOOM_WEIGHT, blur_func)
    total_time = time.time() - initial_time
    print(f"Tempo Gaussiana: {total_time}")
    cv2.imwrite(f"{IMAGE_FOLDER}/01 - bloom_gaussiana.bmp", out_img)

    # Filtro da média
    blur_func = lambda img, sigma: approx_gaussian_blur(img, sigma, NUM_ITER_MEAN)
    initial_time = time.time()
    out_img = bloom(img, THRESHOLD, NUM_BLURS, INIT_SIGMA, BLOOM_WEIGHT, blur_func)
    total_time = time.time() - initial_time
    print(f"Tempo Filtro da Média: {total_time}")
    cv2.imwrite(f"{IMAGE_FOLDER}/02 - bloom_media.bmp", out_img)
