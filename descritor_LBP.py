# coding=utf-8
import numpy
import cv2
from skimage.feature import local_binary_pattern
from skimage import data
import numpy
import time
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import KFold

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
    '''
    return descriptor'''


def mandaDescritor():
    total_treino = []
    total_teste = []

    # gera descritor treino
    for cont in range(0, 1):
        img = cv2.imread("treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png")
        imagem = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t1 = geraDescritor(imagem)
        total_treino.append(t1)
        print ("ESSE E O TREINO: ",  (numpy.array(t1).shape))
        print t1


    # gera decritor teste
    for cont in range(0, 1):
        img = cv2.imread("testes\\train_5a_01" + "{0:03}".format(cont) + ".png")
        imagem = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        t2 = geraDescritor(imagem)
        total_teste.append(t2)
        print ("\n\nESSE E O TESTE:",  (numpy.array(t2).shape))
        print t2

    return total_treino, total_teste

winSize = (128,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

for cont in range(0, 1):
    imagem = cv2.imread("treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png")
    arrayHog = descriptor.compute(imagem)
    print "\n DESCRITOR HOG"
    print numpy.shape(arrayHog)
    print(arrayHog)


# vamos começar a gerar o descritor
listaTeste = []
listaTreino = []

leque = []
print("\nComeçando a leitura")
mandaDescritor()


print("OLHA SO ",)

ee = numpy.asarray(listaTreino, dtype=float)
ii = numpy.asarray(listaTeste, dtype=float)

numpy.array(listaTreino, dtype=float)

print ("\nChamando na rede")
print(ee.shape, ii.shape)

'''
# converte de 3D para um array 2D
# a função fit da biblioteca learn só aceita array 2D

nsamples, nx, ny = listaTreino.shape
novo_treino = listaTreino.reshape((nsamples,nx*ny))

nsamples, nx, ny = listaTeste.shape
novo_teste = listaTeste.reshape((nsamples,nx*ny))

rede.fit(novo_treino, novo_teste)
'''

#rede.fit(ee, ii)

print ("\nFim da rede")