import numpy
import cv2


HOGSLista = []

winSize = (64,64)
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

for cont in range(0, 1000):
    imagem = cv2.imread("treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png")
    HOGSLista.append(descriptor.compute(imagem))

HOGSArray = numpy.array(HOGSLista)
print(HOGSArray.shape)

numpy.save("Treinamento_Z", HOGSArray)

HOGSLista.clear()

for cont in range(0, 1000):
    imagem = cv2.imread("treinamento\\train_53_00" + "{0:03}".format(cont) + ".png")
    HOGSLista.append(descriptor.compute(imagem))

HOGSArray = numpy.array(HOGSLista)

numpy.save("Treinamento_S", HOGSArray)

HOGSLista.clear()

for cont in range(0, 1000):
    imagem = cv2.imread("treinamento\\train_58_00" + "{0:03}".format(cont) + ".png")
    HOGSLista.append(descriptor.compute(imagem))

HOGSArray = numpy.array(HOGSLista)

numpy.save("Treinamento_X", HOGSArray)

HOGSLista.clear()

#Testes


for cont in range(0, 300):
    imagem = cv2.imread("testes\\train_5a_01" + "{0:03}".format(cont) + ".png")
    HOGSLista.append(descriptor.compute(imagem))

HOGSArray = numpy.array(HOGSLista)

numpy.save("Testes_Z", HOGSArray)

HOGSLista.clear()

for cont in range(0, 300):
    imagem = cv2.imread("testes\\train_53_01" + "{0:03}".format(cont) + ".png")
    HOGSLista.append(descriptor.compute(imagem))

HOGSArray = numpy.array(HOGSLista)

numpy.save("Testes_S", HOGSArray)

HOGSLista.clear()

for cont in range(0, 300):
    imagem = cv2.imread("testes\\train_58_01" + "{0:03}".format(cont) + ".png")
    HOGSLista.append(descriptor.compute(imagem))

HOGSArray = numpy.array(HOGSLista)

numpy.save("Testes_X", HOGSArray)