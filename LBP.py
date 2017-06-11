# coding=utf-8
import cv2
from skimage.feature import local_binary_pattern
import numpy
from sklearn.neural_network import MLPClassifier
from LabelLetra import LabelLetra
import time
from datetime import datetime
from sklearn.model_selection import KFold
import pickle

n_points = 16
radius = 2
method = 'uniform'



def rodaTudo():
    '''
    # gera os descritores

    print("\nComeçando a leitura descritor")
    horario_inicio = datetime.now()
    treino_entrada, teste_entrada = mandaDescritor()
    horario_fim = datetime.now()
    print ("Sucesso na leitura")

    # gera arquivo com o tempo
    tempo_descrever_imagens = (
    "\nInicio em: " + horario_inicio.strftime('%d/%m/%Y %H:%M:%S') + "\nFim em: " + horario_fim.strftime(
        '%d/%m/%Y %H:%M:%S'))
    with open("descritores\\lbp\\tempo_descrever_imagens.txt", "a") as myfile:
        myfile.write(tempo_descrever_imagens)
    myfile.close()
    '''


    # le o descritor gerado
    treino_entrada, treino_saida = leitorDescritor()

    # faz as operações na rede
    tempo_inicio = datetime.now()
    controlaRede(treino_entrada, treino_saida)
    tempo_fim = datetime.now()

    # gera arquivo
    tempo_rede = ("\nInicio em: " + tempo_inicio.strftime('%d/%m/%Y %H:%M:%S') + "\nFim em: " + tempo_fim.strftime(
        '%d/%m/%Y %H:%M:%S'))
    with open("descritores\\lbp\\tempo_rede.txt", "a") as myfile:
        myfile.write(tempo_rede)
    myfile.close()

    return


# metodo para gerar o descritor
def geraDescritor(imagem):
    lbp = local_binary_pattern(imagem, n_points, radius, 'uniform')


    (hist, _) = numpy.histogram(lbp.ravel(),
                             bins=numpy.arange(0, n_points + 3),
                             range=(0, n_points + 2))


    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


# metodo para mandar imagens para o descritor
def mandaDescritor():
    entrada_treino = []
    entrada_teste = []
    letras = []

    # region Carrega labels das letras
    labelA = LabelLetra("A", "41")
    letras.append(labelA)

    labelB = LabelLetra("B", "42")
    letras.append(labelB)

    labelC = LabelLetra("C", "43")
    letras.append(labelC)

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

    for letraLab in letras:
        # limpa as listas
        del entrada_treino[:]
        del entrada_teste[:]

        # gera descritor para as imagens treinamento
        for cont in range(0, 1000):
            nomeArquivo = cv2.imread("dataset2\\treinamento\\train_" + letraLab.label + "_00" + "{0:03}".format(cont) + ".png")
            imagem = cv2.cvtColor(nomeArquivo, cv2.COLOR_BGR2GRAY)
            entrada_treino.append(geraDescritor(imagem))

        # salva arquivo com o descritor
        print("Salvando Treinamento " + letraLab.letra + ", tamanho:" + str(len(entrada_treino)))
        numpy.save("descritores\\lbp\\Treinamento_" + letraLab.letra, entrada_treino)
        ''' TEM 300 IMAGENS TESTE PRA CADA LETRA?
        # gera descritor para as imagens teste
        for cont in range(0, 300):
            nomeArquivo = cv2.imread("dataset2\\testes\\train_" + letraLab.label + "_01" + "{0:03}".format(cont) + ".png")
            imagem = cv2.cvtColor(nomeArquivo, cv2.COLOR_BGRA2GRAY)
            entrada_teste.append(geraDescritor(imagem))

        # salva arquivo com o descritor
        print("Salvando Testes " + letraLab.letra + ", tamanho:" + str(len(entrada_teste)))
        numpy.save("descritores\\lbp\\Testes_" + letraLab.letra, entrada_teste)
        '''
    # retorna as listas de descritores geradas
    return entrada_treino, entrada_teste


# metodo para gerar arquivo config do LBP
def geraConfig():
    configtxtdata = ("\nExecucao em " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M"))

    # insere os parametros no arquivo config.txt
    configtxt = ("\n\nLBP (descritor) \nn_points: %s \nradius: %s \nmethod: %s" % (n_points, radius, method))

    with open("descritores\\lbp\\config.txt", "a") as myfile:
        myfile.write(configtxtdata)
        myfile.write(configtxt)
    myfile.close()


# metodo para ler os npy gerados pelo descritor
def leitorDescritor():
    entrada_treino = []
    entrada_teste = []
    saida_treino = []
    saida_teste = []
    letras = []

    # region Carrega labels das letras
    labelA = LabelLetra("A", "41")
    letras.append(labelA)

    labelB = LabelLetra("B", "42")
    letras.append(labelB)

    labelC = LabelLetra("C", "43")
    letras.append(labelC)

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

    for letraLab in letras:
        dados = numpy.load("descritores\\lbp\\Treinamento_" + letraLab.letra + ".npy")
        for ent in dados:
            entrada_treino.append(ent)
            saida_treino.extend(letraLab.letra)

    return entrada_treino, saida_treino


# metodo para rodar a rede
def controlaRede(treino_entrada, treino_saida):
    errortxt = []
    configtxt = []

    # MLPClassifier
    hidden_layer_sizes = (250)
    activation = 'logistic'  # sigmoid
    solver = 'adam'
    alpha = 1e-5
    batch_size = 'auto'
    learning_rate = 'adaptive'
    learning_rate_init = 0.001
    power_t = 0.5
    max_iter = 200
    shuffle = True
    random_state = 20
    tol = 0.0001
    verbose = False
    warm_start = False
    momentum = 0.9
    nesterovs_momentum = True
    early_stopping = False
    validation_fraction = 0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-08

    rede = MLPClassifier(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init,
                         power_t,
                         max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
                         early_stopping, validation_fraction, beta_1, beta_2, epsilon)

    print("Tamanho da lista de Treinamento: " + str(len(treino_entrada)))
    print("Tamanho da lista de saida_treino: " + str(len(treino_saida)))


    # corrige dimensao do array
    treino_entrada = numpy.array(treino_entrada)
    treino_entrada = treino_entrada.reshape(len(treino_entrada), -1)
    treino_saida = numpy.array(treino_saida)

    print("Iniciando Treinamento da rede...")

    entrada_treino = entrada_teste = []
    resposta_treino = resposta_teste = []

    # K FOLD CROSS
    k_fold = KFold(n_splits=5, random_state=None, shuffle=True)
    epoca = 0

    for idTreino, idTeste in k_fold.split(treino_entrada):
        print(" -> rodando epoca: ", epoca)

        # seleciona datasets unicos para treinar e testar

        entrada_treino = (treino_entrada[idTreino])
        resposta_treino = (treino_saida[idTreino])

        entrada_treino =  numpy.array(entrada_treino)
        resposta_treino = numpy.array(resposta_treino)

        rede.fit(entrada_treino, resposta_treino)


        # atualiza epoca
        epoca = epoca + 1

    # gera o model.dat
    pickle.dump(rede, open("descritores\\lbp\\model.dat", "wb"))

    # retorna
    return





rodaTudo()
