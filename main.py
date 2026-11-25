from src.segmentacao import processar_dataset
from src.ocr_tesseract import executar_tesseract
from src.ocr_easyocr import executar_easyocr


def main():
    print("1. Segmentando e extraindo placas de 'placas_exercício'...")
    dados_segmentados = processar_dataset(
        pasta_entrada="placas_exercicio",
        pasta_saida="resultados/placas_segmentadas",
    )
    print(f"   Total de placas extraidas: {len(dados_segmentados)}")

    print("2. Executando OCR com Tesseract...")
    res_tesseract = executar_tesseract(
        dados_segmentados,
        arquivo_saida="resultados/resultados_tesseract.txt",
    )

    print("3. Executando OCR com EasyOCR...")
    res_easyocr = executar_easyocr(
        dados_segmentados,
        arquivo_saida="resultados/resultados_easyocr.txt",
    )

    print("\n=== COMPARATIVO DE RESULTADOS (TESSERACT vs EASYOCR) ===")
    print(f"{'Imagem':<10} | {'Tesseract OCR':<16} | {'EasyOCR':<16}")
    print("-" * 48)
    for img in dados_segmentados:
        print(f"{img:<10} | {res_tesseract[img]:<16} | {res_easyocr[img]:<16}")

    print("\nArquivos gerados em 'resultados/':")
    print(" - resultados/placas_segmentadas/ (recortes das placas)")
    print(" - resultados/resultados_tesseract.txt")
    print(" - resultados/resultados_easyocr.txt")


if __name__ == "__main__":
    main()
