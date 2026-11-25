import os
import re
import easyocr

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


def executar_easyocr(dados_segmentados, arquivo_saida="resultados/resultados_easyocr.txt"):
    """
    Aplica EasyOCR nas placas segmentadas (testando faixa colorida e pre-processadas),
    aplica normalizacao Mercosul e salva os textos reconhecidos em arquivo .txt.
    """
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
    leitor = easyocr.Reader(["en"], gpu=False)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    resultados = {}

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write("RESULTADOS - EASYOCR\n")
        f.write("=" * 65 + "\n")

        for nome_arquivo, item in dados_segmentados.items():
            prep = item["preprocessadas"]
            candidatos = []

            for tecnica in ("faixa_colorida", "clahe_otsu", "otsu", "equalizada"):
                deteccoes = leitor.readtext(
                    prep[tecnica], detail=1, paragraph=False, allowlist=allowlist
                )
                partes = []
                confs = []
                for caixa, texto, conf in deteccoes:
                    limpo = limpar_texto_placa(texto)
                    if limpo:
                        x_esq = min(p[0] for p in caixa)
                        partes.append((x_esq, limpo))
                        confs.append(conf)

                texto_concat = "".join(t for _, t in sorted(partes))
                texto_norm = normalizar_mercosul(texto_concat)
                conf_media = (sum(confs) / len(confs)) if confs else 0.0
                candidatos.append((tecnica, texto_norm, conf_media))

            validos = [c for c in candidatos if PADRAO_PLACA.fullmatch(c[1])]
            if validos:
                escolhido = max(validos, key=lambda c: c[2])
            else:
                nao_vazios = [c for c in candidatos if c[1]]
                escolhido = (
                    min(nao_vazios, key=lambda c: (abs(len(c[1]) - 7), -c[2]))
                    if nao_vazios
                    else ("nenhuma", "NAO_RECONHECIDO", 0.0)
                )

            texto_final = escolhido[1]
            conf_final = escolhido[2]
            formato_ok = bool(PADRAO_PLACA.fullmatch(texto_final))
            resultados[nome_arquivo] = texto_final

            detalhes = " | ".join(f"{tec}: {txt or '-'}" for tec, txt, _ in candidatos)
            f.write(
                f"{nome_arquivo:<8} | Final: {texto_final:<15} | Confianca: {conf_final:.2f} | "
                f"Formato valido: {str(formato_ok):<5} | {detalhes}\n"
            )

    return resultados