"""Biblioteca de funções de processamento de imagens"""
import math
import cv2
import numpy as np
from filters import unsharp

def float_to_uint8(img):
    return (img*255).astype(np.uint8)

def uint8_to_float(img):
    return img.astype(np.float32) / 255

def binarize(img, threshold):
    """Binariza a imagem usando um threshold global"""
    return (img > threshold).astype(float)

def labelize(img, min_width=1, min_height=1, min_n_px=1):
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

def high_value_pass_filter(img, cut_value):
    """Aplica um filtro deixando passar apenas as fontes de luz
    :param img: Imagem de entrada, com apenas uma camada
    :param cut_value: A intensidade de corte
    :returns: A imagem com apenas as fontes de luz
    """
    _, mask = cv2.threshold(img, cut_value, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    img = cv2.copyTo(img, mask)
    return img

def approx_gaussian_blur(img, sigma, num_iter):
    """Aplica um filtro aproximadamente gaussiano usando varios box blur
    :param img: Imagem de entrada
    :param sigma: Desvio padrão da distribuição gaussiana do filtro
    :param num_iter: Quantas vezes aplicar o box blur
    :returns: Imagem filtrada
    """
    # Calcula o tamanho da janela que cria um desvio padrao próximo de sigma
    size = int(np.floor(np.sqrt((sigma*sigma*12 + 1)/num_iter)))
    if not size % 2:
        size += 1
    if size <= 1:
        print(f"Não conseguiu aproximar gaussiana com sigma={sigma} por {num_iter} box blurs")
        return img
    for _ in range(num_iter):
        img = cv2.blur(img, (size, size))
    return img

def bloom(img, threshold, num_blurs, init_sigma, weight, blur_func):
    """Aplica o efeito bloom.
    O efeito bloom é simulado separando as partes brilhantes da imagem,
        borrando elas com sigma dobrando a cada vez
        e somando todas as imagens borradas sobre a original.

    :param img: Imagem de entrada
    :param num_blurs: Quantas camadas de borrado aplicar nas partes brilhantes
    :param weight: Peso com qual a imagem do bloom é somada à original
    imagens borradas
    :param blur_func: A função de borrar imagem com argumentos (img, sigma)
    :returns: A imagem com o efeito bloom
    """
    img = uint8_to_float(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    bright = high_value_pass_filter(img[:, :, 1], threshold)

    bloom = np.zeros(bright.shape, dtype=np.float32)
    sigma = init_sigma
    for _ in range(num_blurs):
        bloom += blur_func(bright, sigma)
        sigma *= 2

    temp = float_to_uint8(np.clip(bright, 0, 1))
    cv2.imwrite("../img/tarefa3/bright.bmp", temp)
    temp = float_to_uint8(np.clip(bloom, 0, 1))
    cv2.imwrite("../img/tarefa3/bloom.bmp", temp)

    out_img = img.copy()
    out_img[:, :, 1] += weight*bloom
    out_img[:, :, 1] = np.clip(out_img[:, :, 1], 0, 1)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_HLS2BGR)
    out_img = float_to_uint8(out_img)
    return out_img

def get_distance_from_mask_max(img, ch, ch_max_value, circular, mask):
    hist_mask = cv2.calcHist([img[:, :, ch]], [0], mask, [ch_max_value+1], [0, ch_max_value+1])
    max_value = np.argmax(hist_mask)
    img = img.astype(float)
    if circular:
        img_ch = img[:, :, ch]
        img_ch += ch_max_value//2 - max_value
        img_ch[img_ch > ch_max_value] -= ch_max_value
        img_ch[img_ch < 0] += ch_max_value
        max_value = ch_max_value//2
    dist = img[:, :, ch] - max_value
    return dist, max_value

def merge_with_thresholds(fg, bg, alpha, low_th, high_th):
    alpha = stretch_normalize(alpha, low_th, high_th, 0, 1)
    alpha_rgb = np.empty(fg.shape)
    alpha_rgb[:, :, 0] = alpha
    alpha_rgb[:, :, 1] = alpha
    alpha_rgb[:, :, 2] = alpha

    fg = cv2.multiply(alpha_rgb, fg.astype(float))
    bg = cv2.multiply(1.0 - alpha_rgb, bg.astype(float))
    result = cv2.add(fg, bg).astype(np.uint8)
    return result

def ln_norm(arrays, n):
    out = np.zeros(arrays[0].shape)
    for arr in arrays:
        out += np.power(arr, n)
    out = np.power(out, 1/n)
    return out

def stretch_normalize(array, low_th, high_th, new_min, new_max):
    array = array.copy()
    array[array < low_th] = low_th
    array[array > high_th] = high_th
    array -= low_th
    array *= (new_max - new_min)/(high_th - low_th)
    array += new_min
    return array

def remove_outliers(img, low_percentile=1, high_percentile=99):
    n_pixels = img.shape[0] * img.shape[1]
    low_percentile = n_pixels * (low_percentile/100)
    high_percentile = n_pixels * (high_percentile/100)

    # Encontra os valores dos extremos
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    sum_ = cv2.integral(hist)[:, 1]
    for px_value, count in enumerate(sum_):
        if count >= low_percentile:
            low_value = px_value
            break
    for px_value in range(len(sum_)-1, -1, -1):
        if sum_[px_value] < high_percentile:
            high_value = px_value
            break
    # Elimina os extremos
    lut = np.arange(256, dtype=np.uint8)
    lut[:low_value] = np.uint8(low_value)
    lut[high_value:] = np.uint8(high_value)
    img = cv2.LUT(img, lut)
    return img.astype(np.uint8), low_value, high_value

def green_screen(fg, bg):
    fg = cv2.GaussianBlur(fg, (3, 3), 0.5)

    hls = cv2.cvtColor(fg, cv2.COLOR_RGB2HLS)

    # Faz uma máscara que vai ter praticamente todo o verde
    green_mask = cv2.inRange(hls[:, :, 0], 50, 70)

    # Calcula a diferença entre o h,l,s do fundo e o h,l,s dos outros pixels
    # Essa "distancia" vai ser a função que avalia
    #   se o pixel faz parte do fundo verde ou nao
    dist_h, max_h = get_distance_from_mask_max(hls, 0, 180, True, green_mask)
    dist_l, max_l = get_distance_from_mask_max(hls, 1, 255, False, green_mask)
    dist_s, max_s = get_distance_from_mask_max(hls, 2, 255, False, green_mask)

    # Como fundo verde costuma ter saturação alta,
    #   dá mais peso pra quando a saturação é menor que a do fundo
    dist_s[dist_s < 0] *= 1.8
    dist_s = np.clip(dist_s, -255, 255)

    dist_h = np.abs(dist_h)
    dist_l = np.abs(dist_l)
    dist_s = np.abs(dist_s)

    dist_h, _, _ = remove_outliers(dist_h.astype(np.uint8), .1, 99.9)
    dist_l, _, _ = remove_outliers(dist_l.astype(np.uint8), .1, 99.9)
    dist_s, _, _ = remove_outliers(dist_s.astype(np.uint8), .1, 99.9)

    # Joga para o intervalo [0, 1], preservando a intensidade das distancias
    dist_h = dist_h.astype(float) / max(max_h, 180-max_h)
    dist_l = dist_l.astype(float) / max(max_l, 255-max_l)
    dist_s = dist_s.astype(float) / max(max_s, 255-max_s)

    # Da peso maior pra distâncias grandes
    th_dist_h = 0.2
    th_dist_l = 0.7
    th_dist_s = 0.75
    dist_h[dist_h > th_dist_h] += (dist_h[dist_h > th_dist_h] - th_dist_h) * 2
    dist_l[dist_l > th_dist_l] += (dist_l[dist_l > th_dist_l] - th_dist_l) * 3
    dist_s[dist_s > th_dist_s] += (dist_s[dist_s > th_dist_s] - th_dist_s) * 2

    # Usa sigmoide pra novamente dar peso maior pra distâncias grandes,
    #   mas também corta pro intervalo [0, 1]
    dist_h = 1 / (1 + np.exp(-3*(dist_h-0.5)))
    dist_l = 1 / (1 + np.exp(-4*(dist_l-0.35)))
    dist_s = 1 / (1 + np.exp(-2*(dist_s-0.6)))

    # Junta as distâncias dando pesos diferentes pros canais
    # O hue é um pouco mais importante,
    #   porque algo com hue muito diferente com certeza não pe verede
    dists = ln_norm((1.2*dist_h, 0.7*dist_l, 0.65*dist_s), 1)
    dists *= 0.6
    dists = np.clip(dists, 0, 1)

    # Junta as duas imagens
    merge = merge_with_thresholds(fg, bg, dists, 0.4, 0.65)
    return merge
