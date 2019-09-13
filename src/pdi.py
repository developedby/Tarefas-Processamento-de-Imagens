"""Biblioteca de funções de processamento de imagens"""
import cv2
import numpy as np

def float_to_uint8(img):
    return (img*255).astype(np.uint8)
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
    inter_img = np.ndarray((img.shape[0], img.shape[1] - wndw[1] + 1, img.shape[2]))
    #horizontal
    for ch in range(img.shape[2]):
        for y in range(img.shape[0]):
            all_win = True
            for x in range(wndw[1]//2, img.shape[1] - wndw[1]//2):
                sum = 0

                if not all_win:
                    sum = prev_sum + img[y, x+(wndw[1]//2), ch] - img[y, x-(wndw[1]//2)-1, ch]
                else:
                    for i in range(wndw[1]):
                        sum += img[y, x-(wndw[1]//2)+i, ch]
                    all_win = False
                inter_img[y, x - wndw[1]//2, ch] = sum
                prev_sum = sum
    out_img = np.ndarray(out_shape)
    #vertical
    for ch in range(inter_img.shape[2]):
        for x in range(inter_img.shape[1]):
            all_win = True
            for y in range(wndw[0]//2, inter_img.shape[0] - wndw[0]//2):
                sum = 0
                if not all_win:
                    sum = prev_sum + inter_img[y + (wndw[0]//2), x, ch] - inter_img[y-(wndw[0]//2)-1, x, ch]
                else:
                    for i in range(wndw[0]):
                        sum += inter_img[y-(wndw[0]//2)+i, x, ch]
                    all_win = False
                out_img[y - wndw[0]//2, x, ch] = sum
                prev_sum = sum

    out_img /= wndw[0]*wndw[1]
    return out_img

def integral_image(img):
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

def mean_filter_integral(img, wndw):
    """Aplica um filtro da média usando uma imagem integrada
    :param img: Imagem de entrada
    :param wndw: Tamanho da janela no formato (height, width)
    :returns: A imagem borrada
    """
    inter_img = integral_image(img)
    out_img = np.ndarray(inter_img.shape)
    r_b = 0
    r_t = 0
    l_b = 0
    l_t = 0
    for ch in range(inter_img.shape[2]):
        for y in range(inter_img.shape[0]):
            for x in range(inter_img.shape[1]):
                if (y+(wndw[0]//2)) >= inter_img.shape[0]:
                    y2 = inter_img.shape[0]-1
                else:
                    y2 = y+(wndw[0]//2)
                if (x+(wndw[1]//2)) >= inter_img.shape[1]:
                    x2 = inter_img.shape[1] - 1
                else:
                    x2 = x+(wndw[1]//2)
                if (x-(wndw[1]//2)-1) < 0:
                    x1 = 1
                else:
                    x1 = x-(wndw[1]//2)
                if (y-(wndw[0]//2)-1) < 0:
                    y1 = 1
                else:
                    y1 = y-(wndw[0]//2)

                l_b = inter_img[y2, x1-1, ch]
                r_b = inter_img[y2, x2, ch]
                l_t = inter_img[y1-1, x1-1, ch]
                r_t = inter_img[y1-1, x2, ch]
                out_img[y, x, ch] = r_b + l_t - l_b - r_t
                out_img[y, x, ch] /= (y2 - y1 + 1) * (x2 - x1 + 1)
    return out_img

def bright_pass_filter(img, threshold):
    """Aplica um filtro deixando passar apenas as fontes de luz
    :param img: Imagem de entrada
    :returns: A imagem com apenas as fontes de luz
    """
    hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Normaliza
    hsl_img = hsl_img.astype(float) / 255
    h, s, l = cv2.split(hsl_img)
    for y in range(s.shape[0]):
        for x in range(s.shape[1]):
            if(s[y, x] < threshold):
                s[y, x] = 0
    out_img = cv2.merge((h, s, l))
    out_img = float_to_uint8(out_img)

    out_img = cv2.cvtColor(out_img, cv2.COLOR_HLS2BGR)
    return out_img

def mean_filters(img, amount):
    """Aplica um filtro da média amount vezes
    :param img: Imagem de entrada
    :param amount: Quantidade que o filtro da média que será aplicado na imagem
    :returns: img contendo a soma de todos os filtros aplicados
    """
    return [img, img]

def bloom(img, threshold, mean_filter_amount, alpha, betta):
    """Aplica o efeito bloom
    :param img: Imagem de entrada
    :param mean_filter_amount: Quantidade imagens borradas que será somada ao
    efeito final
    :param alpha: Valor que será multiplicado a imagem resultante de mean_filters
    :param betta: Valor que será multiplicado a imagem resultante da soma das
    imagens borradas
    :returns: A imagem com o efeito bloom
    """
    out_img = bright_pass_filter(img, threshold)
    return out_img
