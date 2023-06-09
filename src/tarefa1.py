"""Tarefa 1 de processamento digital de imagens
Descrição: Realiza o rotulamento de uma imagem,
    marcando os objetos encontrados com um retângulo verde
Autores: Nicolas Abril e Álefe Felipe Gonçalves Dias
Professor: Bogdan Nassu
Engenharia de Computação UTFPR-CT
08/2019
"""
import time
import cv2
import numpy as np
from pdi import *

# Parâmetros
IMAGE_FOLDER = "../img"
INPUT_IMAGE = "arroz.bmp"
NEGATIVE = False
THRESHOLD = 0.7
MIN_HEIGHT = 2
MIN_WIDTH = 2
MIN_N_PIXELS = 20

if __name__ == '__main__':
    # Abre a imagem
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao abrir a imagem")
        exit()
    # Normaliza
    img = img.astype(float) / 255

    # Inverte
    if NEGATIVE:
        img = invert(img)
        out_img = (img*255).astype(np.uint8)
        cv2.imwrite(f"{IMAGE_FOLDER}/0.5 - negativa.bmp", out_img)

    # Binariza
    img = binarize(img, THRESHOLD)
    out_img = (img*255).astype(np.uint8)
    cv2.imwrite(f"{IMAGE_FOLDER}/01 - binarizada.bmp", out_img)

    # Rotula
    initial_time = time.time()
    components = labelize(img, MIN_WIDTH, MIN_HEIGHT, MIN_N_PIXELS)
    total_time = time.time() - initial_time
    print(f"Tempo: {total_time}")
    print(f"{len(components)} componentes detectados")

    # Desenha os retângulos contornantes
    out_img = (img*255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
    for component in components:
        out_img = draw_bounding_rectangle(out_img, component)
    cv2.imwrite(f"{IMAGE_FOLDER}/02 - out.bmp", out_img)
