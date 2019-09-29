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
IMAGE_FOLDER = "../img/tarefa4"
INPUT_IMAGE = "150.bmp"#"60.bmp"
THRESHOLD = 0.7
NUM_ITER_MEAN = 3
BLOOM_WEIGHT = 0.12
NUM_BLURS = 5
INIT_SIGMA = 5

if __name__ == '__main__':
    # Abre a imagem
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}", cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao abrir a imagem")
        exit()

    h, w, _ = img.shape

    #f_img = uint8_to_float(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (0, 0), INIT_SIGMA)
    #cv2.imwrite(f"{IMAGE_FOLDER}/blur.bmp", blur)
    norma = np.zeros(gray.shape)
    norma = cv2.normalize(gray,  norma, 150, 50, cv2.NORM_MINMAX)
    cv2.imwrite(f"{IMAGE_FOLDER}/norma.bmp", norma)
    sharp = unsharp(norma, 2*((w+h)//100)+1, 3)
    out_img = (sharp*255).astype(np.uint8)
    cv2.imwrite(f"{IMAGE_FOLDER}/sharp.bmp", sharp)
    #thresh = cv2.adaptiveThreshold(sharp, 255,
    #                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY, 77, -1)
    _, thresh = cv2.threshold(sharp, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{IMAGE_FOLDER}/thresh.bmp", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{IMAGE_FOLDER}/open.bmp", open)
    eroded = cv2.erode(open, kernel, iterations=1)
    cv2.imwrite(f"{IMAGE_FOLDER}/eroded.bmp", eroded)

    f_img = uint8_to_float(eroded)
    components = labelize(f_img, 1, 1, 1)
    print(len(components))
