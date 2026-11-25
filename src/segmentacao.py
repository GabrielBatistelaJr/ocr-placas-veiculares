import os
import cv2
import numpy as np


def ordenar_cantos(pontos):
    """Ordena os 4 vertices de um quadrilatero: topo-esq, topo-dir, base-dir, base-esq."""
    pontos = pontos.reshape(4, 2).astype(np.float32)
    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1).ravel()
    return np.array([
        pontos[np.argmin(soma)],
        pontos[np.argmin(diferenca)],
        pontos[np.argmax(soma)],
        pontos[np.argmax(diferenca)],
    ], dtype=np.float32)


def pontuar_candidato_placa(imagem_bgr, poligono):
    """
    Avalia se o quadrilatero retificado corresponde a uma placa veicular:
    verifica a presenca da tarja azul superior e o contraste dos caracteres no centro.
    """
    origem = ordenar_cantos(poligono)
    destino = np.array([[0, 0], [599, 0], [599, 219], [0, 219]], dtype=np.float32)
    matriz = cv2.getPerspectiveTransform(origem, destino)
    recorte = cv2.warpPerspective(imagem_bgr, matriz, (600, 220))

    # Verifica proporcao de pixels azuis na faixa superior da placa
    topo_hsv = cv2.cvtColor(recorte[6:52, 25:575], cv2.COLOR_BGR2HSV)
    mascara_azul = cv2.inRange(topo_hsv, (90, 65, 40), (138, 255, 255))
    proporcao_azul = np.count_nonzero(mascara_azul) / mascara_azul.size

    # Verifica contraste e brilho na faixa central dos caracteres
    centro_cinza = cv2.cvtColor(recorte[60:200, 54:592], cv2.COLOR_BGR2GRAY)
    contraste = float(np.std(centro_cinza)) / 128.0
    brilho_medio = float(np.mean(centro_cinza)) / 255.0

    # Descarta regioes escuras demais, estouradas ou sem contraste de caracteres
    if brilho_medio < 0.25 or brilho_medio > 0.92 or contraste < 0.30:
        return 0.0

    # Exige tarja azul minima ou alto contraste (para placas cinzas antigas sem tarja)
    if proporcao_azul < 0.16 and contraste < 0.48:
        return 0.0

    return proporcao_azul * 4.0 + contraste


def detectar_e_extrair_placa(imagem_bgr):
    """
    Detecta a placa veicular combinando contornos geometricos (com retangulo minimo
    para bordas irregulares) e deteccao da tarja azul Mercosul no espaco HSV.
    """
    altura, largura = imagem_bgr.shape[:2]
    area_img = largura * altura

    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)

    candidatos = []

    # 1. Deteccao por bordas Canny + fechamento morfologico e contornos
    for t1, t2 in [(60, 160), (40, 130), (80, 200)]:
        bordas = cv2.Canny(suavizada, t1, t2)
        bordas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            area = cv2.contourArea(contorno)
            # Limite minimo 0.010 evita mini-adesivos no porta-malas (ex: 19.JPG)
            if not (0.010 * area_img <= area <= 0.45 * area_img):
                continue

            perimetro = cv2.arcLength(contorno, True)
            poligono = None

            for eps in (0.02, 0.03, 0.04, 0.05):
                aprox = cv2.approxPolyDP(contorno, eps * perimetro, True)
                if len(aprox) == 4 and cv2.isContourConvex(aprox):
                    poligono = aprox
                    break

            # Fallback geometrico para placas com borda serrilhada ou colada
            if poligono is None:
                rect = cv2.minAreaRect(contorno)
                w_rect, h_rect = rect[1]
                area_rect = w_rect * h_rect
                if area_rect > 0 and (area / area_rect) >= 0.72:
                    poligono = np.int32(cv2.boxPoints(rect)).reshape(4, 1, 2)

            if poligono is None:
                continue

            x, y, w, h = cv2.boundingRect(poligono)
            proporcao = w / max(h, 1)

            if 2.0 <= proporcao <= 5.5:
                score = pontuar_candidato_placa(imagem_bgr, poligono)
                if score > 0:
                    candidatos.append((score, area, poligono))

    # 2. Deteccao direta pela tarja azul Mercosul no espaco HSV
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    azul = cv2.inRange(hsv, (90, 65, 40), (138, 255, 255))
    azul = cv2.morphologyEx(azul, cv2.MORPH_CLOSE, np.ones((15, 5), np.uint8))
    contornos_azul, _ = cv2.findContours(azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos_azul:
        x, y, w, h = cv2.boundingRect(c)
        if y <= 10 or w < 95 or h < 6 or (w / max(h, 1)) < 3.2:
            continue

        altura_placa = int(w / 2.85)
        y1 = max(0, y - int(h * 0.08))
        y2 = min(altura, y1 + altura_placa)
        x1 = max(0, x - int(w * 0.02))
        x2 = min(largura, x + w + int(w * 0.02))

        poli = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
        area = (x2 - x1) * (y2 - y1)
        if 0.010 * area_img <= area <= 0.45 * area_img:
            score = pontuar_candidato_placa(imagem_bgr, poli)
            if score > 0:
                candidatos.append((score, area, poli))

    # 3. Prioriza o maior score e corrige perspectiva para 600x220
    if candidatos:
        candidatos.sort(key=lambda item: (item[0], item[1]), reverse=True)
        poligono = candidatos[0][2]
    else:
        y1, y2 = int(altura * 0.35), int(altura * 0.75)
        x1, x2 = int(largura * 0.25), int(largura * 0.75)
        poligono = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)

    origem = ordenar_cantos(poligono)
    destino = np.array([[0, 0], [599, 0], [599, 219], [0, 219]], dtype=np.float32)
    matriz = cv2.getPerspectiveTransform(origem, destino)
    placa_corrigida = cv2.warpPerspective(imagem_bgr, matriz, (600, 220))

    imagem_marcada = imagem_bgr.copy()
    cv2.polylines(imagem_marcada, [poligono.reshape(-1, 1, 2)], True, (0, 255, 0), 3)

    return placa_corrigida, imagem_marcada


def preprocessar_placa_para_ocr(placa_bgr):
    """
    Recorta a faixa central dos caracteres (evitando a tarja azul 'BRASIL' e o 'BR')
    e gera diferentes variantes de pre-processamento para testar no OCR.
    """
    faixa_colorida = placa_bgr[58:202, 54:592]
    faixa_cinza = cv2.cvtColor(faixa_colorida, cv2.COLOR_BGR2GRAY)

    # Tecnica 1: Suavizacao Gaussiana + Binarizacao de Otsu
    blur = cv2.GaussianBlur(faixa_cinza, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tecnica 2: Limiarizacao Adaptativa Gaussiana
    adaptativo = cv2.adaptiveThreshold(
        faixa_cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    # Tecnica 3: Equalizacao CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cinza_clahe = clahe.apply(faixa_cinza)
    _, clahe_otsu = cv2.threshold(cinza_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tecnica 4: Equalizacao global de histograma + Otsu (melhora placas muito claras)
    eq_global = cv2.equalizeHist(faixa_cinza)
    _, equalizada = cv2.threshold(eq_global, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "faixa_colorida": faixa_colorida,
        "faixa_cinza": faixa_cinza,
        "otsu": otsu,
        "adaptativo": adaptativo,
        "clahe_otsu": clahe_otsu,
        "equalizada": equalizada,
    }


def processar_dataset(pasta_entrada="placas_exercicio", pasta_saida="resultados/placas_segmentadas"):
    """
    Processa todas as imagens do dataset, salva os recortes das placas segmentadas
    e retorna um dicionario com as variantes pre-processadas para o OCR.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    arquivos = sorted(
        [f for f in os.listdir(pasta_entrada) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda nome: int(os.path.splitext(nome)[0]) if os.path.splitext(nome)[0].isdigit() else nome,
    )

    dados_segmentados = {}

    for nome_arquivo in arquivos:
        caminho = os.path.join(pasta_entrada, nome_arquivo)
        imagem = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)
        if imagem is None:
            continue

        placa, _ = detectar_e_extrair_placa(imagem)
        cv2.imwrite(os.path.join(pasta_saida, f"placa_{nome_arquivo}"), placa)

        dados_segmentados[nome_arquivo] = {
            "placa_bgr": placa,
            "preprocessadas": preprocessar_placa_para_ocr(placa),
        }

    return dados_segmentados