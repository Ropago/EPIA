# coding=utf-8
import cv2
from skimage.feature import local_binary_pattern
import numpy
from sklearn.neural_network import MLPClassifier
from LabelLetra import LabelLetra

# MLPClassifier
hidden_layer_sizes=(500)
activation='logistic' #sigmoid
solver='adam'
alpha=1e-5
batch_size='auto'
learning_rate='adaptive'
learning_rate_init=0.001
power_t=0.5
max_iter=200
shuffle=True
random_state=1
tol=0.0001
verbose=False
warm_start=False
momentum=0.9
nesterovs_momentum=True
early_stopping=False
validation_fraction=0.1
beta_1=0.9
beta_2=0.999
epsilon=1e-08

rede = MLPClassifier(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t,
                     max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
                     early_stopping, validation_fraction, beta_1, beta_2, epsilon)

# sobre o descritor LBP
LBPLista = []
n_points = 16
radius = 2


# gera o descritor
def geraDescritor(imagem):
    # calcula o descritor
    lbp = local_binary_pattern(imagem, n_points, radius, 'uniform')


    (hist, _) = numpy.histogram(lbp.ravel(),
                             bins=numpy.arange(0, n_points + 3),
                             range=(0, n_points + 2))


    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    # return histograma
    return hist


def mandaDescritor():
    entrada_treino = []
    entrada_teste = []
    saida_treino = []
    saida_teste = []

    # gera descritor treino
    for cont in range(0, 100):
        img = cv2.imread("dataset2\\treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png")
        imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t1 = geraDescritor(imagem)
        entrada_treino.append(t1)
        saida_treino.append("Z")
        print "ESSE E O TREINO: ",  (numpy.array(t1).shape)
        print "\n", t1, "\n"

    return entrada_treino, saida_treino


def novoMandaDescritor():
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
    for letraLab in letras:
        print "Lendo a letra: ", letraLab.letra
        # gera o descritor para as imagens de treinamento
        for cont in range(0, 1000):
            nomeArquivo = cv2.imread("dataset2\\treinamento\\train_" + letraLab.label + "_00" + "{0:03}".format(cont) + ".png")
            imagem = cv2.cvtColor(nomeArquivo, cv2.COLOR_BGR2GRAY)
            entrada_treino.append(geraDescritor(imagem))
            entrada_teste.append(letraLab.letra)

    return entrada_treino, entrada_teste



print("\nComeçando a leitura")
treino_entrada, treino_saida = novoMandaDescritor()

print treino_saida

sucesso = numpy.array(treino_saida)
felicidade = numpy.array(treino_entrada)
treino_entrada = felicidade.reshape(len(felicidade), -1)
print felicidade.shape, sucesso.shape


rede.fit(felicidade, sucesso)
'''
for cada in treino_entrada:
    num = treino_entrada.index(cada)
    item  = treino_entrada[num]
    otem = treino_saida[num]
    print "\n", "ITEM E OTEM: ", item, " ** ", otem
    item = numpy.asarray(item)
    otem = numpy.array(otem)
    print item.shape, "_________", otem.shape
    rede.fit(item, otem.all(axis= None))
'''


print ("\nFim da rede")