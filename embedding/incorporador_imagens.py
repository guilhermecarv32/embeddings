import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions, RunningMode

from colorama import Back, Style

MODELO = "C:\Users\Aluno\Desktop\embedding\modelos\mobilenet_v3_large.tflite"

GATOS_BRANCOS = [
    "C:\Users\Aluno\Desktop\embedding\imagens\gato_branco1.jpg",
    "C:\Users\Aluno\Desktop\embedding\imagens\gato_branco2.jpg"
]

GATOS_PRETOS = [
    "C:\Users\Aluno\Desktop\embedding\imagens\gato_preto1.jpg",
    "C:\Users\Aluno\Desktop\embedding\imagens\gato_preto2.jpg"
]

CACHORROS = [
    "C:\Users\Aluno\Desktop\embedding\imagens\cachorro1.jpg",
    "C:\Users\Aluno\Desktop\embedding\imagens\cachorro2.jpg"
]

BICHOS_PARA_TESTES = {
    {
        "tipo": "cachorro",
        "imagem": "C:\Users\Aluno\Desktop\embedding\imagens\cachorro_teste.jpg"
    },
    {
        "tipo": "gato branco",
        "imagem": "C:\Users\Aluno\Desktop\embedding\imagens\gato_branco_teste.jpg"
    },
    {
        "tipo": "gato preto",
        "imagem": "C:\Users\Aluno\Desktop\embedding\imagens\gato_preto_teste.jpg"
    }
}

def configurar():
    configurado, incorporador = False, None

    try:
        opcoes = ImageEmbedderOptions(base_options=BaseOptions(model_asset_path=MODELO), quantize = True, running_mode=RunningMode.IMAGE)
        incorporador = ImageEmbedder.create_from_options(opcoes)

        configurado = True
    except Exception as e:
        print(f"ocorreu um erro configurando incorporador: {str(e)}")

    return configurado, incorporador

def processar(imagem, incorporador):
    processada, incorporacao = False, None

    try:
        imagem = mp.Image.create_from_file(imagem)
        incorporacao = incorporador.embed(imagem)

        processada = True
    except Exception as e:
        print(f"ocorreu um erro processando a imagem: {str(e)}")

    return processada, incorporacao

def processar_bichos(imagens, incorporador):
    processados, incorporacoes = False, []

    for imagem in imagens:
        processada, incorporacao = processar(imagem, incorporador)
        if processada:
            incorporacoes.append(incorporacao)

    return processados, incorporacoes

def comparar(imagem, grupo_candidato, incorporador):
    similaridade = 0.0

    processada, incorporacao = processar(imagem, incorporador)
    if processada:
        for individuo in grupo_candidatos:
            nova_similaridade = incorporador.cosine_similarity(incorporacao.embeddings[0], individuo.embeddings[0])
            similaridade = nova_similaridade if nova_similaridade > similaridade else similaridade
    else:
        print(f"não foi possível comparar o bicho {bicho['tipo']} através de sua imagem: {bicho['imagem']}")

    return similaridade

if __name__ == "__main__":
    configurado, incorporador = configurar()

    if configurado:
        _, gatos_pretos = processar_bichos(GATOS_PRETOS, incorporador)
        _, gatos_brancos = processar_bichos(GATOS_BRANCOS, incorporador)
        _, cachorros = processar_bichos(CACHORROS, incorporador)

        for bicho in BICHOS_PARA_TESTES:
            print(f"testando similaridade entre {bicho['tipo']} e gatos brancos")
            similaridade_com_gatos_brancos = comparar(bicho['imagem'], gatos_brancos, incorporador)
            print(f"distancias do bicho {bicho['tipo']} a gatos brancos: {similaridade_com_gatos_brancos}")
            
            print(f"testando similaridade entre {bicho['tipo']} e gatos pretos")
            similaridade_com_gatos_pretos = comparar(bicho['imagem'], gatos_pretos, incorporador)
            print(f"distancias do bicho {bicho['tipo']} a gatos pretos: {similaridade_com_gatos_pretos}")

            print(f"testando similaridade entre {bicho['tipo']} e cachorros")
            similaridade_com_cachorros = comparar(bicho['imagem'], cachorros, incorporador)         
            print(f"distancias do bicho {bicho['tipo']} a cachorros: {similaridade_com_cachorros}")

            if similaridade_com_gatos_brancos > similaridade_com_gatos_pretos and similaridade_com_gatos_brancos > similaridade_com_cachorros:
                print(Back.CYAN + f"o bicho considerado {bicho['tipo']} é mais similar a gatos brancos" + Style.RESET_ALL)
            elif similaridade_com_gatos_pretos > similaridade_com_gatos_brancos and similaridade_com_gatos_pretos > similaridade_com_cachorros:
                print(Back.RED + f"o bicho considerado {bicho['tipo']} é mais similar a gatos pretos" + Style.RESET_ALL)
            else:
                print(Back.YELLOW + f"o bicho consideado {bicho['tipo']} é mais similar um cachorros" + Style.RESET_ALL)
