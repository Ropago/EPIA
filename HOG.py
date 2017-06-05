import numpy
import cv2
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import KFold


def leitor():

    HOGSLista = []

    winSize = (128, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                   histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    configtxtdata = ("\nExecucao em " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M"))

    # insere os parametros no arquivo config.txt
    configtxt = ("\n\nHOG (descritor) \nwinSize: %s \nblockSize: %s \nblockStride: %s \ncellSize: %s \nnbins: %s \n"
                 "derivAperture: %s \nwinSigma: %s \nhistogramNormType: %s \nL2HysThreshold: %s \ngammaCorrection: %s \n"
                 "nlevels: %s" % (winSize, blockSize, blockStride,
                                  cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold,
                                  gammaCorrection, nlevels))

    with open("config.txt", "a") as myfile:
        myfile.write(configtxtdata)
        myfile.write(configtxt)
    myfile.close()

    print("Começando a leitura")

    for cont in range(0, 1000):
        imagem = cv2.imread("treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png")
        HOGSLista.append(descriptor.compute(imagem))

    print("Salvando Treinamento Z, tamanho:" + str(len(HOGSLista)))
    numpy.save("Treinamento_Z", HOGSLista)
    print("Salvo")

    del HOGSLista[:]

    for cont in range(0, 1000):
        imagem = cv2.imread("treinamento\\train_53_00" + "{0:03}".format(cont) + ".png")
        HOGSLista.append(descriptor.compute(imagem))

    print("Salvando Treinamento S, tamanho:" + str(len(HOGSLista)))
    numpy.save("Treinamento_S", HOGSLista)
    print("Salvo")

    del HOGSLista[:]

    for cont in range(0, 1000):
        imagem = cv2.imread("treinamento\\train_58_00" + "{0:03}".format(cont) + ".png")
        HOGSLista.append(descriptor.compute(imagem))

    print("Salvando Treinamento X, tamanho:" + str(len(HOGSLista)))
    numpy.save("Treinamento_X", HOGSLista)
    print("Salvo")

    del HOGSLista[:]

    # Testes


    for cont in range(0, 300):
        imagem = cv2.imread("testes\\train_5a_01" + "{0:03}".format(cont) + ".png")
        HOGSLista.append(descriptor.compute(imagem))

    print("Salvando Testes Z, tamanho:" + str(len(HOGSLista)))
    numpy.save("Testes_Z", HOGSLista)
    print("Salvo")

    del HOGSLista[:]

    for cont in range(0, 300):
        imagem = cv2.imread("testes\\train_53_01" + "{0:03}".format(cont) + ".png")
        HOGSLista.append(descriptor.compute(imagem))

    print("Salvando Testes S, tamanho:" + str(len(HOGSLista)))
    numpy.save("Testes_S", HOGSLista)
    print("Salvo")

    del HOGSLista[:]

    for cont in range(0, 300):
        imagem = cv2.imread("testes\\train_58_01" + "{0:03}".format(cont) + ".png")
        HOGSLista.append(descriptor.compute(imagem))

    print("Salvando Testes X, tamanho:" + str(len(HOGSLista)))
    numpy.save("Testes_X", HOGSLista)
    print("Salvo")

    del HOGSLista[:]


def treinador():
    HOGS = numpy.load("Treinamento_S.npy")
    print("Lendo arquivos de Treinamento S. Tamanho: " + str(len(HOGS)))

    HOGX = numpy.load("Treinamento_X.npy")
    print("Lendo arquivos de Treinamento X. Tamanho: " + str(len(HOGX)))

    HOGZ = numpy.load("Treinamento_Z.npy")
    print("Lendo arquivos de Treinamento Z. Tamanho: " + str(len(HOGZ)))

    entradas = []
    respostas = []
    errortxt = []
    configtxt = []

    MLPClassifier
    hidden_layer_sizes = (500)
    activation = 'logistic'  # sigmoid
    solver = 'adam'
    alpha = 1e-5
    batch_size = 'auto'
    learning_rate = 'adaptive'
    learning_rate_init = 0.001
    power_t = 0.5
    max_iter = 200
    shuffle = True
    random_state = 1
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

    errortxt.append("\nExecucao em " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M"))

    # adiciona ao config.txt
    configtxt.append(
        "\n MLPClassifier \n hidden_layer_sizes : %s\n activation : %s\n solver : %s\n alpha : %s\n batch_size : %s\n"
        " learning_rate : %s\n learning_rate_init : %s\n power_t : %s\n max_iter : %s\n shuffle : %s\n "
        "random_state : %s\n tol : %s\n verbose : %s\n warm_start : %s\n momentum : %s\n nesterovs_momentum : %s\n"
        "early_stopping : %s\n validation_fraction : %s\n beta_1 : %s\n beta_2 : %s\n epsilon : %s" %

        (str(hidden_layer_sizes), str(activation), str(solver), str(alpha), str(batch_size), str(learning_rate),
         str(learning_rate_init), str(power_t), str(max_iter), str(shuffle), str(random_state), str(tol),
         str(verbose), str(warm_start), str(momentum), str(nesterovs_momentum), str(early_stopping),
         str(validation_fraction), str(beta_1), str(beta_2), str(epsilon)))

    with open("config.txt", "a") as myfile:
        for item in configtxt:
            myfile.write(item)
    myfile.close()

    print("Unificando as listas de Treinamento...")
    for ent in HOGS:
        entradas.append(ent)
        respostas.extend("S")

    for ent in HOGX:
        entradas.append(ent)
        respostas.extend("X")

    for ent in HOGZ:
        entradas.append(ent)
        respostas.extend("Z")

    print("Pronto")

    print("Tamanho da lista de Treinamento: " + str(len(entradas)))
    print("Tamanho da lista de respostas: " + str(len(respostas)))

    print("Corrigindo dimensao da lista de entradas...")
    EntradasArray = numpy.array(entradas)
    ArrayCorrigida = EntradasArray.reshape(len(EntradasArray), -1)

    respostas = numpy.array(respostas)
    print("Pronto")

    # treinamento

    print("Iniciando Treinamento da rede...")

    TempoInicio = time.time()

    # K FOLD CROSS
    k_fold = KFold(n_splits=5, random_state=None, shuffle=True)
    epoca = 0

    for idTreino, idTeste in k_fold.split(ArrayCorrigida):
        print(" -> rodando epoca: ", epoca)

        # seleciona datasets unicos para treinar e testar
        entrada_treino, entrada_teste = ArrayCorrigida[idTreino], ArrayCorrigida[idTeste]
        resposta_treino, resposta_teste = respostas[idTreino], respostas[idTeste]

        print("shape 1: %s  shape 2: %s" % (entrada_treino, resposta_treino))

        print("Tamanhos:\nEntrada: %s\nRespostas: %s" % (str(len(entrada_treino)), str(len(resposta_treino))))

        print(" -> treinando a rede")
        # treina rede
        rede.fit(entrada_treino, resposta_treino)

        print(" -> fazendo previsão")
        prediz = rede.predict(entrada_teste)

        print(" -> rodando o treinamento para pegar o erro")
        retesta = rede.predict(entrada_treino)

        print("Erro medio ", 1 - rede.score(entrada_treino, resposta_treino))

        # salva info no array de error.txt
        errortxt.append(str(epoca) + ";" + str(1 - rede.score(entrada_treino, resposta_treino)) + ";" + str(
            1 - rede.score(entrada_teste, resposta_teste)))

        # atualiza epoca
        epoca = epoca + 1

    # gera o model.dat
    pickle.dump(rede, open("model.dat", "wb"))

    TempoFim = time.time()

    from sklearn.metrics import classification_report, confusion_matrix

    print("\n MATRIZ DE CONFUSAO")
    print(confusion_matrix(resposta_teste, prediz))
    print("\n CLASSIFICACAO")
    print(classification_report(resposta_teste, prediz))

    print("Rede treinada em " + str(TempoFim - TempoInicio) + " segundos")

    file = open("error.txt", "w")
    for item in errortxt:
        file.write("%s\n" % item)
    file.close()


leitor()

treinador()