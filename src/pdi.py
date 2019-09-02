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
        for pixel in component:
            y_min = min(y_min, pixel[0])
            y_max = max(y_max, pixel[0])
            x_min = min(x_min, pixel[1])
            x_max = max(x_max, pixel[1])
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        if (width >= min_width
                and height >= min_height
                and len(component) >= min_n_px):
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
    """Desenha na imagem os retângulos que contornam os componentes"""
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

def mean_filter_bad(img, wndw):
    """Aplica um filtro da média usando algoritmo ingênuo
    :param img: Imagem de entrada
    :param wndw: Tamanho da janela no formato (height, width)
    :returns: A imagem borrada
    """
    out_shape = (img.shape[0] -  wndw[0] + 1, img.shape[1] - wndw[1] + 1, img.shape[2])
    out_img = np.ndarray(out_shape)
    for ch in range(img.shape[2]):
        for y in range(wndw[0]//2, img.shape[0] - wndw[0]//2):
            for x in range(wndw[1]//2, img.shape[1] - wndw[1]//2):
                sum = 0
                for i in range(wndw[0]):
                    for j in range(wndw[1]):
                        sum += img[y-(wndw[0]//2)+i, x-(wndw[1]//2)+j, ch]
                #print(x, y)
                out_img[y-wndw[0]//2, x-wndw[1]//2, ch] = sum
    out_img /= wndw[0] * wndw[1]
    return out_img

def mean_filter_separable(img, wndw):
    """Aplica um filtro da média usando algoritmo separável aproveitando soma
    :param img: Imagem de entrada
    :param wndw: Tamanho da janela no formato (height, width)
    :returns: A imagem borrada
    """
    out_shape = (img.shape[0] -  wndw[0] + 1, img.shape[1] - wndw[1] + 1, img.shape[2])
    inter_img = np.ndarray(out_shape)
    #horizontal
    for ch in range(img.shape[2]):
        for y in range(wndw[0]//2, img.shape[0] - wndw[0]//2):
            all_win = True
            for x in range(wndw[1]//2, img.shape[1] - wndw[1]//2):
                sum = 0

                if not all_win:
                    sum = prev_sum + img[y, x+(wndw[1]//2), ch] - img[y, x-(wndw[1]//2)-1, ch]
                else:
                    for i in range(wndw[1]):
                        sum += img[y, x-(wndw[1]//2)+i, ch]
                    all_win = False
                inter_img[y - wndw[0]//2, x - wndw[1]//2, ch] = sum
                prev_sum = sum
    out_img = np.ndarray(out_shape)
    #vertical
    for ch in range(inter_img.shape[2]):
        for x in range(wndw[1]//2, inter_img.shape[1] - wndw[1]//2):
            all_win = True
            for y in range(wndw[0]//2, inter_img.shape[0] - wndw[0]//2):
                sum = 0
                if not all_win:
                    sum = prev_sum + inter_img[y + (wndw[0]//2), x, ch] - inter_img[y-(wndw[0]//2)-1, x, ch]
                else:
                    for i in range(wndw[0]):
                        sum += inter_img[y-(wndw[0]//2)+i, x, ch]
                    all_win = False
                out_img[y - wndw[0]//2, x - wndw[1]//2, ch] = sum
                prev_sum = sum

    out_img /= wndw[0]*wndw[1]
    return out_img

def integrated_image(img):
    """Transforma a imagem em uma imagem integral
    :param img: Imagem de entrada
    :returns: A imagem integrada
    """
    out_img = np.ndarray(img.shape)
    #esquerda
    for ch in range(img.shape[2]):
        for y in range(img.shape[0]):
            out_img[y, 0, ch] = img[y, 0, ch]
            for x in range(1, img.shape[1]):
                out_img[y, x, ch] = img[y, x, ch] + out_img[y, x-1, ch]
    #acima
    for ch in range(out_img.shape[2]):
        for y in range(1, out_img.shape[0]):
            for x in range(out_img.shape[1]):
                out_img[y, x, ch] = out_img[y, x, ch] + out_img[y-1, x, ch]
    return out_img

def mean_filter_integrated(img, wndw):
    """Aplica um filtro da média usando uma imagem integrada
    :param img: Imagem de entrada
    :param wndw: Tamanho da janela no formato (height, width)
    :returns: A imagem borrada
    """
    inter_img = integrated_image(img)
    out_img = np.ndarray(inter_img.shape)
    r_b = 0
    r_t = 0
    l_b = 0
    l_t = 0
    for ch in range(inter_img.shape[2]):
        for y in range(inter_img.shape[0]):
            for x in range(inter_img.shape[1]):
                if (y+(wndw[0]//2)+1) >= inter_img.shape[0]:
                    if (x+(wndw[1]//2)+1) >= inter_img.shape[1]:
                        r_b = inter_img[inter_img.shape[0]-1, inter_img.shape[1]-1, ch]
                    else:
                        r_b = inter_img[inter_img.shape[0]-1, x+(wndw[1]//2)+1, ch]
                    if (x-(wndw[1]//2)-1) < 0:
                        l_b = inter_img[inter_img.shape[0]-1, 0, ch]
                    else:
                        l_b = inter_img[inter_img.shape[0]-1, x-(wndw[1]//2)-1, ch]
                else:
                    if (x+(wndw[1]//2)+1) >= inter_img.shape[1]:
                        r_b = inter_img[y+(wndw[0]//2)+1, inter_img.shape[1]-1, ch]
                    else:
                        r_b = inter_img[y+(wndw[0]//2)+1, x+(wndw[1]//2)+1, ch]
                    if (x-(wndw[1]//2)-1) < 0:
                        l_b = inter_img[y+(wndw[0]//2)+1, 0, ch]
                    else:
                        l_b = inter_img[y+(wndw[0]//2)+1, x-(wndw[1]//2)-1, ch]
                if (y-(wndw[0]//2)-1) < 0:
                    if (x+(wndw[1]//2)+1) >= inter_img.shape[1]:
                        r_t = inter_img[0, inter_img.shape[1]-1, ch]
                    else:
                        r_t = inter_img[0, x+(wndw[1]//2)+1, ch]
                    if (x-(wndw[1]//2)-1) < 0:
                        l_t = inter_img[0, 0, ch]
                    else:
                        l_t = inter_img[0, x-(wndw[1]//2)-1, ch]
                else:
                    if (x+(wndw[1]//2)+1) >= inter_img.shape[1]:
                        r_t = inter_img[y-(wndw[0]//2)-1, inter_img.shape[1]-1, ch]
                    else:
                        r_t = inter_img[y-(wndw[0]//2)-1, x+(wndw[1]//2)+1, ch]
                    if (x-(wndw[1]//2)-1) < 0:
                        l_t = inter_img[y-(wndw[0]//2)-1, 0, ch]
                    else:
                        l_t = inter_img[y-(wndw[0]//2)-1, x-(wndw[1]//2)-1, ch]
                out_img[y, x, ch] = r_b + l_t - l_b - r_t
    out_img /= wndw[0] * wndw[1]
    return out_img
