import os
import re
import pytesseract

PADRAO_PLACA = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")

PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
PARA_NUMERO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}


def limpar_texto_placa(texto):
    """Remove espacos/simbolos e mantem apenas caracteres alfanumericos maiusculos."""
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def normalizar_mercosul(texto):
    """
    Remove eventuais vazamentos de 'BR'/'BRASIL' e corrige caracteres
    confundidos visualmente de acordo com a posicao no padrao Mercosul (LLLNLNN).
    """
    limpo = limpar_texto_placa(texto)
    if len(limpo) > 7 and limpo.startswith("BR"):
        limpo = limpo[2:]
    if len(limpo) > 7:
        limpo = re.sub(r"BRAS.*$", "", limpo)
    if len(limpo) != 7:
        return limpo

    chars = list(limpo)
    for idx in (0, 1, 2, 4):
        chars[idx] = PARA_LETRA.get(chars[idx], chars[idx])
    for idx in (3, 5, 6):
        chars[idx] = PARA_NUMERO.get(chars[idx], chars[idx])
    return "".join(chars)


def executar_tesseract(dados_segmentados, arquivo_saida="resultados/resultados_tesseract.txt"):
    """
    Aplica Tesseract OCR nas diferentes variantes pre-processadas de cada placa,
    aplica normalizacao Mercosul, seleciona o melhor resultado e grava o relatorio.
    """
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
    alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={alfabeto}"

    resultados = {}

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write("RESULTADOS - TESSERACT OCR\n")
        f.write("=" * 65 + "\n")

        for nome_arquivo, item in dados_segmentados.items():
            prep = item["preprocessadas"]
            leituras = {}

            for tecnica in ("otsu", "adaptativo", "clahe_otsu", "equalizada", "faixa_cinza"):
                bruto = pytesseract.image_to_string(prep[tecnica], config=config)
                leituras[tecnica] = normalizar_mercosul(bruto)

            # Prioriza leitura no padrao Mercosul (7 caracteres); senao, mais proxima de 7
            validos = [t for t in leituras.values() if PADRAO_PLACA.fullmatch(t)]
            if validos:
                melhor = validos[0]
            else:
                nao_vazios = [t for t in leituras.values() if t]
                melhor = min(nao_vazios, key=lambda t: abs(len(t) - 7)) if nao_vazios else "NAO_RECONHECIDO"

            formato_ok = bool(PADRAO_PLACA.fullmatch(melhor))
            resultados[nome_arquivo] = melhor

            f.write(
                f"{nome_arquivo:<8} | Final: {melhor:<15} | Formato valido: {str(formato_ok):<5} | "
                f"Otsu: {leituras['otsu'] or '-'} | Adapt: {leituras['adaptativo'] or '-'} | "
                f"CLAHE: {leituras['clahe_otsu'] or '-'} | Eq: {leituras['equalizada'] or '-'}\n"
            )

    return resultados