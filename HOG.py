import numpy
import cv2
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import KFold
from LabelLetra import LabelLetra


def leitor():

    HOGSLista = []

    winSize = (128, 128)
    blockSize = (16, 16)
    blockStride = (16, 16)
    cellSize = (16, 16)
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

    letras = []

    #region Carrega labels das letras
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

    #endregion

    for letraLab in letras:

        for cont in range(0, 1000):
            nomeArquivo = "treinamento\\train_" + letraLab.label + "_00" + "{0:03}".format(cont) + ".png"
            imagem = cv2.imread(nomeArquivo)
            HOGSLista.append(descriptor.compute(imagem))

        print("Salvando Treinamento " + letraLab.letra + ", tamanho:" + str(len(HOGSLista)))
        EntradasArray = numpy.array(HOGSLista)
        ArrayCorrigida = EntradasArray.reshape(len(EntradasArray), -1)
        numpy.save("Treinamento_"+ letraLab.letra, ArrayCorrigida)

        del HOGSLista[:]

        for cont in range(0, 300):
            imagem = cv2.imread("testes\\train_" + letraLab.label + "_01" + "{0:03}".format(cont) + ".png")
            HOGSLista.append(descriptor.compute(imagem))

        print("Salvando Testes " + letraLab.letra + ", tamanho:" + str(len(HOGSLista)))
        EntradasArray = numpy.array(HOGSLista)
        ArrayCorrigida = EntradasArray.reshape(len(EntradasArray), -1)
        numpy.save("Testes_" + letraLab.letra, ArrayCorrigida)

        del HOGSLista[:]


def treinador():

    entradas = []
    respostas = []

    '''
    HOGS = numpy.load("Treinamento_S.npy")
    print("Lendo arquivos de Treinamento S. Tamanho: " + str(len(HOGS)))

    HOGX = numpy.load("Treinamento_X.npy")
    print("Lendo arquivos de Treinamento X. Tamanho: " + str(len(HOGX)))

    HOGZ = numpy.load("Treinamento_Z.npy")
    print("Lendo arquivos de Treinamento Z. Tamanho: " + str(len(HOGZ)))
    
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
    '''
    letras = []

    #region Carrega labels das letras
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

    #endregion

    for letraLab in letras:

        dados = numpy.load("Treinamento_" + letraLab.letra + ".npy")

        for ent in dados:
            entradas.append(ent)
            respostas.extend(letraLab.letra)


    errortxt = []
    configtxt = []

    MLPClassifier
    hidden_layer_sizes = (250)
    activation = 'logistic'  # sigmoid
    solver = 'adam'
    alpha = 1e-5
    batch_size = 'auto'
    learning_rate = 'adaptive'
    learning_rate_init = 0.01
    power_t = 0.5
    max_iter = 200
    shuffle = True
    random_state = 20
    tol = 0.0001
    verbose = True
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


    print("Tamanho da lista de Treinamento: " + str(len(entradas)))
    print("Tamanho da lista de respostas: " + str(len(respostas)))

    print("Corrigindo dimensao da lista de entradas...")
    EntradasArray = numpy.array(entradas)

    respostas = numpy.array(respostas)
    print("Pronto")

    # treinamento

    print("Iniciando Treinamento da rede...")

    TempoInicio = time.time()

    # K FOLD CROSS
    k_fold = KFold(n_splits=5, random_state=None, shuffle=True)
    epoca = 0

    for idTreino, idTeste in k_fold.split(EntradasArray):
        print(" -> rodando epoca: ", epoca)

        # seleciona datasets unicos para treinar e testar
        entrada_treino, entrada_teste = EntradasArray[idTreino], EntradasArray[idTeste]
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


#leitor()

treinador()