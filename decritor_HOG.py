# coding=utf-8
import numpy
import cv2


HOGSLista = []

winSize = (128,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (16,16)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

# insere os parametros no arquivo config.txt
configtxt = ("\nHOG (descritor) \n winSize: %s,blockSize: %s,blockStride: %s,cellSize: %s,nbins: %s,derivAperture: %s,winSigma: %s,"
             " histogramNormType: %s, L2HysThreshold: %s,gammaCorrection: %s,nlevels: %s" % (winSize,blockSize,blockStride,
             cellSize, nbins, derivAperture, winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels))

with open("config.txt", "a") as myfile:
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

#Testes


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