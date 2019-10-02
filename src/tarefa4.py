"""Tarefa 4 de processamento digital de imagens
Descrição: Conta arrozes com diferentes imagens
Autores: Nicolas Abril e Álefe Felipe Gonçalves Dias
Professor: Bogdan Nassu
Engenharia de Computação UTFPR-CT
10/2019
"""
import time
import cv2
import numpy as np
from pdi import unsharp, labelize, uint8_to_float

# Parâmetros
IMAGE_FOLDER = "../img/tarefa4"
INPUT_IMAGE = "60.bmp"

UNSHARP_SIGMA_DIVIDER = 200
UNSHARP_WEIGHT = 1
THRESHOLD = 240
AREA_CUTOFF = 0.2
AREA_CORRECTION = [0.02, 0.01, 0.002]

if __name__ == '__main__':
    # Abre a imagem
    img = cv2.imread(f"{IMAGE_FOLDER}/{INPUT_IMAGE}", cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao abrir a imagem")
        exit()
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    norma = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    sharp = unsharp(norma,
                    2*(max(w, h)//UNSHARP_SIGMA_DIVIDER)+1,
                    UNSHARP_WEIGHT)
    cv2.imwrite(f"{IMAGE_FOLDER}/1-sharp.bmp", sharp)

    _, thresh = cv2.threshold(sharp, THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{IMAGE_FOLDER}/2-thresh.bmp", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open_ = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{IMAGE_FOLDER}/3-open.bmp", open_)

    f_img = uint8_to_float(open_)
    components = labelize(f_img, 1, 1, 1)

    areas = np.array(list(map(len, components)))
    median = np.median(areas)
    rel_areas = areas / median

    corrected_areas = rel_areas[rel_areas > AREA_CUTOFF]
    poly_correction = np.zeros(corrected_areas.shape)
    for order, weight in enumerate(AREA_CORRECTION):
        order_n_correction = np.power(corrected_areas, order+1)*weight
        poly_correction += order_n_correction
    corrected_areas += poly_correction
    corrected_areas = np.round(corrected_areas).astype(int)

    num_rice = 0
    for area in corrected_areas:
        if area < 1:
            num_rice += 1
        else:
            num_rice += int(round(area))
    print(num_rice)
