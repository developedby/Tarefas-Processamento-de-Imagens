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
from pdi import unsharp, labelize, uint8_to_float

# Parâmetros
IMAGE_FOLDER = "../img"
INPUT_IMAGE = "60.bmp"
THRESHOLD = 0.7
NUM_ITER_MEAN = 3
BLOOM_WEIGHT = 0.12
NUM_BLURS = 5
INIT_SIGMA = 1.2

if __name__ == '__main__':
    # Abre a imagem
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}", cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao abrir a imagem")
        exit()

    h, w, _ = img.shape

    #f_img = uint8_to_float(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = unsharp(gray, 2*((w+h)//100)+1, 0.5)
    thresh = cv2.adaptiveThreshold(sharp, 1,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 2*((w+h)//100)+1, -0.1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    eroded = cv2.erode(open, kernel, iterations=2)
    components = labelize(eroded, 2, 2, 4)
    print(len(components))
