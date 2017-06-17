# coding=utf-8
import cv2
from skimage.feature import local_binary_pattern
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from LabelLetra import LabelLetra
import time
from datetime import datetime
from sklearn.model_selection import KFold
import pickle
import os
import codecs
import matplotlib.pyplot as plt
import itertools


# configurações do descritor
radius = 3
n_points = 8 * radius
method = 'uniform'

# configurações da rede
# MLPClassifier: configurações da rede
hidden_layer_sizes = (18)
activation = 'logistic'  # sigmoid
solver = 'sgd'
alpha = 1e-5
batch_size = 'auto'
learning_rate = 'adaptive'
learning_rate_init = 0.1
power_t = 0.5
max_iter = 1000
shuffle = True
random_state = 20
tol = 0.0001
verbose = True
warm_start = False
momentum = 0.9
nesterovs_momentum = False
early_stopping = True
validation_fraction = 0.1
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08

# configurações da pasta principal
pasta_origem = "descritores\\lbp\\"


def rodaTudo():

    # gera os descritores
    '''
    print("\nComeçando a leitura descritor")
    horario_inicio = datetime.now()
    treino_entrada, teste_entrada = geraDescritor()
    horario_fim = datetime.now()
    print ("Sucesso na leitura")

    # gera arquivo com o tempo
    tempo_descrever_imagens = (
    "\nInicio em: " + horario_inicio.strftime('%d/%m/%Y %H:%M:%S') + "\nFim em: " + horario_fim.strftime(
        '%d/%m/%Y %H:%M:%S'))
    with open(pasta_origem + "tempo_descrever_imagens.txt", "a") as myfile:
        myfile.write(tempo_descrever_imagens)
    myfile.close()
    '''


    # le o descritor gerado
    treino_entrada, teste_entrada, treino_saida, teste_saida = leitorDescritor()

    # faz as operações na rede
    tempo_inicio = datetime.now()
    controlaRede(treino_entrada, teste_entrada, treino_saida, teste_saida)
    tempo_fim = datetime.now()

    # gera arquivo
    tempo_rede = ("\nInicio em: " + tempo_inicio.strftime('%d/%m/%Y %H:%M:%S') + "\nFim em: " + tempo_fim.strftime(
        '%d/%m/%Y %H:%M:%S'))
    with open(pasta_origem + "tempo_rede.txt", "a") as myfile:
        myfile.write(tempo_rede)
    myfile.close()

    return


# metodo para criar array de todas as classes de imagens
def geraArrayLetras():
    letras = []
    # region Carrega labels das letras
    labelA = LabelLetra("A", "41")
    letras.append(labelA)

    labelB = LabelLetra("B", "42")
    letras.append(labelB)

    labelC = LabelLetra("C", "43")
    letras.append(labelC)
    '''
    labelD = LabelLetra("D", "44")
    letras.append(labelD)

    labelE = LabelLetra("E", "45")
    letras.append(labelE)
    
    labelF = LabelLetra("F", "46")
    letras.append(labelF)

    labelG = LabelLetra("G", "47")
    letras.append(labelG)

    labelH = LabelLetra("H", "48")
    letras.append(labelH)

    labelI = LabelLetra("I", "49")
    letras.append(labelI)

    labelJ = LabelLetra("J", "4a")
    letras.append(labelJ)

    labelK = LabelLetra("K", "4b")
    letras.append(labelK)

    labelL = LabelLetra("L", "4c")
    letras.append(labelL)

    labelM = LabelLetra("M", "4d")
    letras.append(labelM)

    labelN = LabelLetra("N", "4e")
    letras.append(labelN)

    labelO = LabelLetra("O", "4f")
    letras.append(labelO)

    labelP = LabelLetra("P", "50")
    letras.append(labelP)

    labelQ = LabelLetra("Q", "51")
    letras.append(labelQ)

    labelR = LabelLetra("R", "52")
    letras.append(labelR)

    labelS = LabelLetra("S", "53")
    letras.append(labelS)

    labelT = LabelLetra("T", "54")
    letras.append(labelT)

    labelU = LabelLetra("U", "55")
    letras.append(labelU)

    labelV = LabelLetra("V", "56")
    letras.append(labelV)

    labelW = LabelLetra("W", "57")
    letras.append(labelW)

    labelX = LabelLetra("X", "58")
    letras.append(labelX)

    labelY = LabelLetra("Y", "59")
    letras.append(labelY)

    labelZ = LabelLetra("Z", "5a")
    letras.append(labelZ)
    '''
    return letras


# metodo para calcular o descritor LBP
def calculaDescritor(imagem):
    lbp = local_binary_pattern(imagem, n_points, radius, method)

    (hist, _) = numpy.histogram(lbp.ravel(), bins=numpy.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    # normaliza o histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist


# metodo para gerar os descritores das imagens
def geraDescritor():
    entrada_treino = []
    entrada_teste = []
    letras = geraArrayLetras()

    # para cada classe de letras gera o descritor de treinamento e teste
    for letraLab in letras:
        # gera descritor para as imagens treinamento
        for cont in range(0, 1000):
            nomeArquivo = cv2.imread("dataset2\\treinamento\\train_" + letraLab.label + "_00" + "{0:03}".format(cont) + ".png")
            imagem = cv2.cvtColor(nomeArquivo, cv2.COLOR_BGR2GRAY)
            entrada_treino.append(calculaDescritor(imagem))

        print("Salvando Treinamento " + letraLab.letra + ", tamanho:" + str(len(entrada_treino)))
        numpy.save(pasta_origem + "Treinamentos_" + letraLab.letra, entrada_treino)

        del entrada_treino[:]

        for cont in range(0, 300):
            nomeArquivo = cv2.imread("dataset2\\testes\\train_" + letraLab.label + "_01" + "{0:03}".format(cont) + ".png")
            imagem = cv2.cvtColor(nomeArquivo, cv2.COLOR_BGR2GRAY)
            entrada_teste.append(calculaDescritor(imagem))

        print("Salvando Testes " + letraLab.letra + ", tamanho:" + str(len(entrada_teste)))
        numpy.save(pasta_origem + "Testes_" + letraLab.letra, entrada_teste)

        del entrada_teste[:]


    # retorna as listas de descritores geradas
    return entrada_treino, entrada_teste


# metodo para ler os npy gerados pelo descritor
def leitorDescritor():
    entrada_treino = []
    entrada_teste = []
    saida_treino = []
    saida_teste = []
    letras = geraArrayLetras()

    for letraLab in letras:
        dados = numpy.load(pasta_origem + "Treinamentos_" + letraLab.letra + ".npy")
        for ent in dados:
            entrada_treino.append(ent)
            saida_treino.extend(letraLab.letra)

    for letraLab in letras:
        dados = numpy.load(pasta_origem + "Testes_" + letraLab.letra + ".npy")
        for ent in dados:
            entrada_teste.append(ent)
            saida_teste.extend(letraLab.letra)

    return entrada_treino, entrada_teste, saida_treino, saida_teste


# metodo para rodar a rede
def controlaRede(treino_entrada, teste_entrada, treino_saida, teste_saida):
    redes = []

    for cont in range(0, 5):
        redes.append(MLPClassifier(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init,
                         power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum,
                         nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon))


    # corrige dimensao do array de treinamento
    treino_entrada = numpy.array(treino_entrada)
    treino_entrada = treino_entrada.reshape(len(treino_entrada), -1)
    treino_saida = numpy.array(treino_saida)

    # corrige dimensao do array de teste
    teste_entrada = numpy.array(teste_entrada)
    teste_entrada = teste_entrada.reshape(len(teste_entrada), -1)
    teste_saida = numpy.array(teste_saida)

    print("Iniciando Treinamento da rede...")

    # TREINA A REDE: kfold com 5 épocas
    k_fold = KFold(n_splits=5, random_state=None, shuffle=True)
    epoca = 0
    erro_validacao = []
    lista_acuracia = []
    erro_treinamento = []

    for idTreino, idTeste in k_fold.split(treino_entrada):
        print(" -> rodando epoca: ", epoca)

        # seleciona datasets unicos para treinar e testar
        entrada_treino = []
        entrada_teste = []
        entrada_treino, entrada_teste = treino_entrada[idTreino], treino_entrada[idTeste]
        resposta_treino, resposta_teste = treino_saida[idTreino], treino_saida[idTeste]

        # converte esses datasets para arrays
        entrada_treino = numpy.array(entrada_treino)
        resposta_treino = numpy.array(resposta_treino)

        # treina a rede: gera o erro de treinamento
        redes[epoca].fit(entrada_treino, resposta_treino, pasta_origem + "error" + str(epoca) +".txt")

        # prediz a rede: gera o erro de validação
        erro_validacao.append(redes[epoca].score(entrada_teste, resposta_teste))

        # armazena acuracia e erro de treinamento
        lista_acuracia.append(redes[epoca].score(teste_entrada, teste_saida))
        erro_treinamento.append(redes[epoca].loss_)

        # atualiza epoca
        epoca = epoca + 1

    #gera arquivos para comparar as redes
    melhorScore = 0.0
    piorScore = 1.0
    melhorRede = 0
    piorRede = 0

    for cont in range(0,5):
        if(erro_validacao[cont] < piorScore):
            piorScore = erro_validacao[cont]
            piorRede = cont
        if(erro_validacao[cont] > melhorScore):
            melhorScore = erro_validacao[cont]
            melhorRede = cont

    print("Resultado das redes:\nA melhor rede é a " + str(melhorRede) + " com pontuacao de " + str(melhorScore))
    print("A pior rede é a " + str(piorRede) +" com pontuacao de " + str(piorScore))


    # TESTA A REDE: com a partição de testes
    print("\nTestando a rede com  a particao de testes...")


    print ("Acuracia da melhor rede: ", lista_acuracia[melhorRede])
    print ("Acuracia da pior rede: ", lista_acuracia[piorRede])


    # gera artefatos
    gera_arquivo_model(redes[melhorRede], isMelhor=True)
    gera_arquivo_model(redes[piorRede], isMelhor=False)
    gera_arquivo_config()
    gera_arquiv_relatorio(lista_acuracia, erro_treinamento, erro_validacao)



    # imprime erro de validação
    print ("Erro de validacao da melhor rede eh: ", erro_validacao[melhorRede])
    print ("Erro de validacao da pior rede eh: ", erro_validacao[piorRede])


    # matriz de confusão para o melhor caso
    matriz_confusao = confusion_matrix(teste_saida, redes[melhorRede].predict(teste_entrada))
    plot_confusion_matrix(matriz_confusao, normalize=False, isMelhor=True)
    plot_confusion_matrix(matriz_confusao, normalize=True, isMelhor=True)

    # matriz de confusão para o pior caso
    matriz_confusao = confusion_matrix(teste_saida, redes[piorRede].predict(teste_entrada))
    plot_confusion_matrix(matriz_confusao, normalize=False, isMelhor=False)
    plot_confusion_matrix(matriz_confusao, normalize=True, isMelhor=False)

    # retorna
    return



# gera arquivo model.dat
def gera_arquivo_model(rede, isMelhor):
    matriz0 = numpy.zeros((rede.hidden_layer_sizes, 577))
    matriz1 = numpy.zeros((rede.n_outputs_, (rede.hidden_layer_sizes + 1)))

    for cont in range(0, rede.hidden_layer_sizes):
        matriz0[cont][0] = rede.intercepts_[0][cont]

    for x in range(0, rede.hidden_layer_sizes):
        for y in range (1, 577):
            matriz0[x][y] = rede.coefs_[0][y-1][x]

    for cont in range(0, rede.n_outputs_):
        matriz1[cont][0] = rede.intercepts_[1][cont]

    for x in range(0, rede.n_outputs_):
        for y in range (1, rede.hidden_layer_sizes + 1):
            matriz1[x][y] = rede.coefs_[1][y-1][x]

    if (isMelhor == True):
        nomeArquivo = "modelMelhor.dat"
    else:
        nomeArquivo = "modelPior.dat"

    pickle.dump((matriz0, matriz1), open(pasta_origem + nomeArquivo, "wb"))
    print("- salva model.dat")


# gera arquivo config.txt
def gera_arquivo_config ():

    configtxtdata = ("Execucao em " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M"))

    # os parametros do descritor
    confighog = ("\n\nHOG (descritor) \nwinSize: %s \nblockSize: %s \nblockStride: %s \ncellSize: %s \nnbins: %s \n"
                 "derivAperture: %s \nwinSigma: %s \nhistogramNormType: %s \nL2HysThreshold: %s \ngammaCorrection: %s \n"
                 "nlevels: %s" % (winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                  histogramNormType, L2HysThreshold, gammaCorrection, nlevels))


    # os parametros da rede
    configrede = (
    "\n\nMLPClassifier (rede)\nhidden_layer_sizes : %s\nactivation : %s\nsolver : %s\nalpha : %s\nbatch_size : %s\n"
    "learning_rate : %s\nlearning_rate_init : %s\npower_t : %s\nmax_iter : %s\nshuffle : %s\n"
    "random_state : %s\ntol : %s\nverbose : %s\nwarm_start : %s\nmomentum : %s\nnesterovs_momentum : %s\n"
    "early_stopping : %s\nvalidation_fraction : %s\nbeta_1 : %s\nbeta_2 : %s\nepsilon : %s" %
    (str(hidden_layer_sizes), str(activation), str(solver), str(alpha), str(batch_size), str(learning_rate),
     str(learning_rate_init), str(power_t), str(max_iter), str(shuffle), str(random_state), str(tol),
     str(verbose), str(warm_start), str(momentum), str(nesterovs_momentum), str(early_stopping),
     str(validation_fraction), str(beta_1), str(beta_2), str(epsilon)))

    try:
        os.remove(pasta_origem + "config.txt")
    except OSError:
        pass

    with codecs.open(pasta_origem + "config.txt", "a", "utf-8") as myfile:
        myfile.write(configtxtdata)
        myfile.write(confighog)
        myfile.write(configrede)
    myfile.close()
    print ("- salva config.txt")

    return


# gera txts para relatorio
def gera_arquiv_relatorio(lista_acuracia, erro_treinamento, erro_validacao):

    # salva erro treinamento
    try:
        os.remove(pasta_origem + "erro_treinamento.txt")
    except OSError:
        pass
    with codecs.open(pasta_origem + "erro_treinamento.txt", "a", "utf-8") as myfile:
        for cont in range(0, 5):
            myfile.write(str(erro_treinamento[cont]) + "\n")
    myfile.close()
    print ("- salva erro_treinamento.txt")


    # salva acuracia media
    try:
        os.remove(pasta_origem + "lista_acuracia.txt")
    except OSError:
        pass
    with codecs.open(pasta_origem + "lista_acuracia.txt", "a", "utf-8") as myfile:
        for cont in range(0,5):
            myfile.write(str(lista_acuracia[cont])  + "\n")
    myfile.close()
    print ("- salva lista_acuracia.txt")


    # salva erro validação
    try:
        os.remove(pasta_origem + "erro_validacao.txt")
    except OSError:
        pass
    with codecs.open(pasta_origem + "erro_validacao.txt", "a", "utf-8") as myfile:
        for cont in range(0,5):
            myfile.write(str(erro_validacao[cont])  + "\n")
    myfile.close()
    print ("- salva erro_validacao.txt")
    return


# plota a matriz de confusão
# retirado de: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, normalize, isMelhor, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    letras = geraArrayLetras()
    classes = []
    for letraLab in letras:
        classes.append(letraLab.letra)

    if (isMelhor == True):
        extra = "[MELHOR CASO]"
    else:
        extra = "[PIOR CASO]"
        cmap = plt.cm.Oranges

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    titulo = "Matriz de confusao normalizada"
    if (normalize ==  False):
        titulo = "Matriz de confusao"

    plt.title(titulo)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.subplots_adjust(bottom=0.15)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, numpy.newaxis]
        nomeArquivo = "matriz_confusao_normalizada"+extra+".png"
    else:
        nomeArquivo = "matriz_confusao"+extra+".png"

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    #plt.tight_layout()
    plt.ylabel("Classe esperada")
    plt.xlabel("Classe prevista")
    fig.savefig(pasta_origem + nomeArquivo)



rodaTudo()
