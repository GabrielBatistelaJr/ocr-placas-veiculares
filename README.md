# Reconhecimento de Placas Veiculares (Tesseract OCR vs EasyOCR)

Pipeline de Processamento de Imagens e Visão Computacional para detecção, segmentação, correção de perspectiva e reconhecimento óptico de caracteres (OCR) de placas veiculares brasileiras (padrão Mercosul), comparando **Tesseract OCR** e **EasyOCR**.

## Estrutura do Projeto

```text
ocr-placas-veiculares/
├── placas_exercício/              # Dataset com as 20 imagens de entrada (1.JPG a 20.JPG)
├── src/
│   ├── __init__.py
│   ├── segmentacao.py             # Detecção por bordas/contornos, correção de perspectiva e pré-processamento
│   ├── ocr_tesseract.py           # Reconhecimento de caracteres com Tesseract OCR
│   └── ocr_easyocr.py             # Reconhecimento de caracteres com EasyOCR
├── resultados/                    # Gerado automaticamente ao rodar main.py
│   ├── placas_segmentadas/        # Recortes das placas extraídas
│   ├── resultados_tesseract.txt   # Textos extraídos pelo Tesseract
│   └── resultados_easyocr.txt     # Textos extraídos pelo EasyOCR
├── main.py                        # Orquestra todo o pipeline e imprime o comparativo
├── requirements.txt               # Dependências Python
└── README.md                      # Documentação e análise comparativa
```

## Como Executar

1. **Instale as dependências Python:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Certifique-se de ter o Tesseract OCR instalado no sistema operacional** (no Windows, caso não esteja no `PATH`, informe o caminho do executável em `src/ocr_tesseract.py`).
3. **Execute o pipeline completo:**
   ```bash
   python main.py
   ```

## Etapas do Pipeline

1. **Detecção e Segmentação (`src/segmentacao.py`):**
   - Conversão para escala de cinza e suavização Gaussiana;
   - Detecção de bordas com Canny, dilatação morfológica e aproximação poligonal (`approxPolyDP`) filtrada por proporção e área (com fallback pela segmentação da tarja azul superior no espaço HSV);
   - Correção de perspectiva (`getPerspectiveTransform` + `warpPerspective`) e recorte da faixa central dos caracteres;
   - Testes de pré-processamento: binarização de Otsu, limiarização adaptativa Gaussiana e equalização de histograma adaptativa (CLAHE + Otsu).
2. **OCR e Geração de Resultados (`src/ocr_tesseract.py` e `src/ocr_easyocr.py`):**
   - Aplicação do Tesseract (`--oem 3 --psm 7` com whitelist alfanumérica) e do EasyOCR (`allowlist` e ordenação espacial das caixas);
   - Validação pelo padrão de placa (`^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$`) e exportação para `.txt`.

## Análise Comparativa (Tesseract vs EasyOCR)

- **Tesseract OCR:** Apresenta execução muito rápida, mas depende fortemente da qualidade da binarização (Otsu, Adaptativa ou CLAHE) e é mais sensível a marcas d'água, sombras diagonais e baixo contraste nos caracteres.
- **EasyOCR:** Baseado em redes neurais profundas, lida melhor com variações de iluminação, sombras e texturas de fundo diretamente sobre a faixa colorida ou com contraste equalizado, porém possui tempo de processamento maior em CPU.
