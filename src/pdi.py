"""Biblioteca de funções de processamento de imagens"""
import cv2
import numpy as np

def binarize(img, threshold):
    """Binariza a imagem usando um threshold global"""
    return (img > threshold).astype(float)

def labelize(img, min_width, min_height, min_n_px):
    """Rotula os componentes de uma imagem binária
    
    :param img: Imagem binária que vai ser rotulada
    :param min_width: Largura mínima de cada componente
    :param min_height: Altura mínima de cada componente
    :param min_n_px: Número mínimo de pixels de cada componente
    :returns: Uma lista com os componentes encontrados (conjunto de pontos)
    """
    img_label = np.copy(img)
    img_label[img_label == 1] = -1
    label = 1
    components = []
    # Encontra os componentes
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img_label[y, x] == -1:
                img_label, component = flood_fill_stack(img, img_label, label, x, y)
                components.append(component)
                label += 1
    # Filtra apenas os componentes válidos
    valid_components = []
    for component in components:
        y_min, x_min = img.shape
        y_max, x_max = 0, 0
        n_pixels = 0
        for pixel in component:
            n_pixels += 1
            y_min = min(y_min, pixel[0])
            y_max = max(y_max, pixel[0])
            x_min = min(x_min, pixel[1])
            x_max = max(x_max, pixel[1])
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        if width >= min_width and height >= min_height and n_pixels >= min_n_px:
            valid_components.append(component)
    return valid_components

def flood_fill_stack(img, img_label, label, x, y):
    """Marca o componente em x, y de uma imagem binária com o valor label"""
    img_label[y, x] = label
    stack = [(y, x)]
    component = []
    while stack:
        (y, x) = stack.pop()
        component.append((y, x))
        if (y > 0
                and img_label[y-1, x] == -1
                and img[y-1, x] == img[y, x]):
            img_label[y-1, x] = label
            stack.append((y-1, x))
        if (y < img.shape[0]-1
                and img_label[y+1, x] == -1
                and img[y+1, x] == img[y, x]):
            img_label[y+1, x] = label
            stack.append((y+1, x))
        if (x > 0
                and img_label[y, x-1] == -1
                and img[y, x-1] == img[y, x]):
            img_label[y, x-1] = label
            stack.append((y, x-1))
        if (x < img.shape[1]-1
                and img_label[y, x+1] == -1
                and img[y, x+1] == img[y, x]):
            img_label[y, x+1] = label
            stack.append((y, x+1))
    return img_label, component

def invert(img):
    """Retorna uma imagem com intensidade invertida"""
    return 1 - img

def draw_bounding_rectangle(img, component, thickness=2):
    y_min, x_min, _ = img.shape
    y_max, x_max = 0, 0
    n_pixels = 0
    for pixel in component:
        n_pixels += 1
        y_min = min(y_min, pixel[0])
        y_max = max(y_max, pixel[0])
        x_min = min(x_min, pixel[1])
        x_max = max(x_max, pixel[1])
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness)
    return img