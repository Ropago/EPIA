# coding=utf-8
import cv2
import cPickle
import numpy

''' Explicação mais detalhada:
 http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
 ------------------------
Python: cv2.SURF([hessianThreshold[, nOctaves[, nOctaveLayers[, extended[, upright]]]]]) → <SURF object>

Parameters:	
hessianThreshold – Threshold for hessian keypoint detector used in SURF.
nOctaves – Number of pyramid octaves the keypoint detector will use.
nOctaveLayers – Number of octave layers within each octave.
extended – Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).
upright – Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).
-----------------------------------
Python: cv2.SURF.detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]]) → keypoints, descriptors
-----------------------------------
Python: cv2.drawKeypoints(image, keypoints[, outImage[, color[, flags]]]) → outImage

Parameters:	
image – Source image.
keypoints – Keypoints from the source image.
outImage – Output image. Its content depends on the flags value defining what is drawn in the output image. See possible flags bit values below.
color – Color of keypoints.
flags – Flags setting drawing features. Possible flags bit values are defined by DrawMatchesFlags. See details above in drawMatches() .
'''
surf = cv2.SURF(400)
surf.extended = False #pra img ser 64 como no descritor HOG no momento. true para 128

# pensar sobre a orientação. É necessário?
# surf.upright = True

SURFlista = []

for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png", 0)
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona os pontos chave na SURFlista
    SURFlista.append(ponto_chave)
    # precisamos serializar os pontos chave dos dados
    index = []
    for point in ponto_chave:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
            point.class_id)
        index.append(temp)

SURFArray = index
numpy.save("Treinamento_sZ", SURFArray)
del SURFlista[:]




for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("treinamento\\train_53_00" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona os pontos chave na SURFlista
    SURFlista.append(ponto_chave)
    # precisamos serializar os pontos chave dos dados
    index = []
    for point in ponto_chave:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
            point.class_id)
        index.append(temp)

SURFArray = index
numpy.save("Treinamento_sS", SURFArray)
del SURFlista[:]





for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("treinamento\\train_58_00" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona os pontos chave na SURFlista
    SURFlista.append(ponto_chave)
    # precisamos serializar os pontos chave dos dados
    index = []
    for point in ponto_chave:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
            point.class_id)
        index.append(temp)

SURFArray = index
numpy.save("Treinamento_sX", SURFArray)
del SURFlista[:]




# -------------------------------------------------------------
# Testes


for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("testes\\train_5a_01" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, )
    # adiciona os pontos chave na SURFlista
    SURFlista.append(ponto_chave)
    # precisamos serializar os pontos chave dos dados
    index = []
    for point in ponto_chave:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
            point.class_id)
        index.append(temp)

SURFArray = index
numpy.save("Testes_sZ", SURFArray)
del SURFlista[:]




for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("testes\\train_53_01" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona os pontos chave na SURFlista
    SURFlista.append(ponto_chave)
    # precisamos serializar os pontos chave dos dados
    index = []
    for point in ponto_chave:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
            point.class_id)
        index.append(temp)

SURFArray = index
numpy.save("Testes_sS", SURFArray)
del SURFlista[:]





for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("testes\\train_58_01" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona os pontos chave na SURFlista
    SURFlista.append(ponto_chave)
    # precisamos serializar os pontos chave dos dados
    index = []
    for point in ponto_chave:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
            point.class_id)
        index.append(temp)

SURFArray = index
numpy.save("Testes_sX", SURFArray)
del SURFlista[:]