"""Biblioteca com alguns filtros de imagem"""
import numpy as np
import cv2

SLICE_TOP = 'top'
SLICE_BOTTOM = 'bottom'

# pylint: disable=invalid-name

def gamma_correction(img, gamma):
    """Aplica uma correção de gamma na imagem
    O valor de cada pixel após a transformação é Vout = 255*(Vin/255)^gamma
    Deixa os pixels mais escuros mais claros, enquanto não muda muito os que já eram claros.

    :param img: Imagem que vai ser corrigida
    :param gamma: Fator exponencial a ser aplicado. Menor deixa mais brilhante
    :returns: A imagem tranformada
    """
    lut = np.arange(256)
    lut = np.uint8(np.clip(pow(lut / 255.0, gamma) * 255.0, 0, 255))
    return cv2.LUT(img, lut)

def linear_correction(img, a=5, b=10, auto=True):
    """Aplica a transformação a(x - b) em cada pixel da imagem.
    Faz com que o fundo fique preto enquanto as partes interessantes ficam mais claras.

    :param img: Imagem que vai ser corrigida
    :param a: Fator que multiplica
    :param b: Fator que subtrai
    :param auto: Se calcula 'a' e 'b' automaticamente ou não
    :returns: A imagem corrigida, 'a' e 'b'
    """
    if auto:
        h, w = img.shape
        bright = img[(h//2)-(h//8):(h//2)+(h//8), w//8:w-w//8]  # Parte que é assumida como clara
        dark = img[:w//8, w//8:w-w//8]  # Parte que é assumida como escura (meio e baixo)
        # 'b' tem que ser menor que o objeto e maior que o fundo
        b = (np.mean(dark) + np.mean(bright)) / 2
        # 'a' tem que trazer toda a borda do objeto para um nivel adequado de brilho
        a = 75.0 / (np.mean(bright) + np.std(bright) - b)

    lut = np.arange(256, dtype=np.int16)
    lut = np.uint8(np.clip(a*(lut - b), 0, 255))
    return cv2.LUT(img, lut), a, b

def normalize_remove_outliers(img, low_percentile=1, high_percentile=99):
    """Normaliza uma imagem para o range [0, 255] removendo extremidades do histograma
    As extremidades são colocadas em 0 e 255
    :param img: Imagem a ser normalizada. Deve ter 1 canal e ter tipo np.uint8
    :param low_percentile: Percentil de pixels que representa o extremo inferior
    :param high_percentile: Percentil de pixels que representa o extremo superior
    :returns: A imagem normalizada, os dois valores de corte
    """
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
    # Normaliza
    img = cv2.normalize(img, img, 255, 0, cv2.NORM_MINMAX)
    return img.astype(np.uint8), low_value, high_value

def unflare(img, strength=2.5, slice_size=0.1, slice_pos=SLICE_BOTTOM):
    """
    Remove um brilho horizontal da imagem
    Pega uma fatia da imagem que vai representar o gradiente

    :param img: Imagem de onde remover o flare
    :param strength: Intensidade do unflare.
        Se for muito baixo não consegue remover completamente.
        Se for muito alto escurece demais.
    :param slice_size: Quanto da imagem vai pegar para a fatia. De 0 a 1.
    :param slice_pos: Se tira uma fatia de baixo ou de cima
    :returns: A imagem corrigida
    """
    # Pega valores invalidos
    if strength <= 0:
        return None
    if slice_size <= 0 or slice_size >= 1:
        return None
    if slice_pos not in (SLICE_BOTTOM, SLICE_TOP):
        return None

    h, w = img.shape

    # Corta uma fatia
    if slice_pos == SLICE_BOTTOM:
        slice_ = img[int(h*(1-slice_size)):h, :]
    elif slice_pos == SLICE_TOP:
        slice_ = img[:int(h*slice_size), :]
    sl_h, _ = slice_.shape

    # Extende a fatia pra toda a imagem
    flare = np.zeros(img.shape, np.uint8)
    for i in range(int(h/sl_h)):
        flare[i*sl_h:(i+1)*sl_h, :] = slice_
    flare[int(h-sl_h):h, :] = slice_

    # Borra a imagem, pra remover ruidos e saltos nas junções das fatias
    k_height = int(2*h*slice_size) if int(2*h*slice_size) % 2 else int(2*h*slice_size) + 1
    k_width = w//100 if (w//100) % 2 else w//100 + 1
    flare = cv2.GaussianBlur(flare, (k_width, k_height), 0)
    # Joga para valores entre 1 e strength, representando a atenuação de cada pixel
    flare = cv2.normalize(flare.astype('float'), None, 1.0, strength, cv2.NORM_MINMAX)
    # Remove o brilho
    unflared = np.uint8(img / flare)
    return unflared

def unsharp(img, sigma, weight):
    """Aplica o filtro unsharp para realçar as bordas da imagem
    Esse filtro subtrae uma versão borrada da original, removendo as baixas frequencias
    out = (1 + weight)*img - weight*blur(img, blur_strength)

    :param img: A imagem a ser filtrada
    :param sigma: Tamanho do blur a ser feito. Maior aumenta a frequência de corte
    :param weight: Peso da subtração da imagem borrada. Maior é mais forte
    :returns: A imagem filtrada
    """
    return cv2.addWeighted(img, 1+weight, cv2.GaussianBlur(img, (0, 0), sigma), -weight, 0)

def richardson_lucy_deconvolution(img, num_iter, psf):
    """Aplica uma deconvolução usando o algoritmo de Richardson-Lucy"""
    # TODO: Testar
    img = img.astype(float)
    psf = psf.astype(float)
    estimate = np.copy(img)
    psf_reversed = psf[::-1, ::-1]
    for _ in range(num_iter):
        estimate_conv = cv2.filter2D(estimate, -1, psf)
        error_estimation = cv2.filter2D(img/estimate_conv, -1, psf_reversed)
        estimate = estimate * error_estimation
    return estimate
