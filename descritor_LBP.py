# coding=utf-8
import cv2
from skimage.feature import local_binary_pattern
import numpy
from sklearn.neural_network import MLPClassifier

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
    descriptor = local_binary_pattern(imagem, n_points, radius, 'uniform')

    # normaliza o histograma
    from scipy.stats import itemfreq
    fator = itemfreq(descriptor.ravel())
    histograma = fator[:, 1]/sum(fator[:, 1])
    # return histograma
    return histograma


def mandaDescritor():
    entrada_treino = []
    entrada_teste = []
    saida_treino = []
    saida_teste = []

    # gera descritor treino
    for cont in range(0, 1):
        img = cv2.imread("dataset2\\treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png")
        imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t1 = geraDescritor(imagem)
        entrada_treino.append(t1)
        saida_treino.append("Z")
        print "ESSE E O TREINO: ",  (numpy.array(t1).shape)
        print "\n", t1, "\n"


    # gera decritor teste
    for cont in range(0, 1):
        img = cv2.imread("dataset2\\testes\\train_5a_01" + "{0:03}".format(cont) + ".png")
        imagem = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        t2 = geraDescritor(imagem)
        entrada_teste.append(t2)
        saida_teste.append("Z")
        print "ESSE E O TESTE:",  (numpy.array(t2).shape)
        print "\n", t2

    return entrada_treino, entrada_teste, saida_treino, saida_teste




print("\nComeçando a leitura")
treino_entrada, teste_entrada, treino_saida, teste_saida = mandaDescritor()



rede.fit(treino_entrada, treino_saida)

print ("\nFim da rede")