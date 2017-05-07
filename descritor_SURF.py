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
surf.extended = True #pra img ser 64 como no descritor HOG no momento. true para 128

# pensar sobre a orientação. É necessário?
# surf.upright = True

SURFlista = []

for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("treinamento\\train_5a_00" + "{0:03}".format(cont) + ".png", 0)
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona o descritor na SURFlista
    SURFlista.append(descritor)

print("Salvando descritor Treinamento surf-Z, tamanho:" + str(len(SURFlista)))
numpy.save("Treinamento_surf-Z", SURFlista)
print("Salvo")
del SURFlista[:]




for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("treinamento\\train_53_00" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona o descritor na SURFlista
    SURFlista.append(descritor)

print("Salvando descritor Treinamento surf-S, tamanho:" + str(len(SURFlista)))
numpy.save("Treinamento_surf-S", SURFlista)
print("Salvo")
del SURFlista[:]




for cont in range(0, 1000):
    # montamos a imagem
    imagem = cv2.imread("treinamento\\train_58_00" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona o descritor na SURFlista
    SURFlista.append(descritor)

print("Salvando descritor Treinamento surf-X, tamanho:" + str(len(SURFlista)))
numpy.save("Treinamento_surf-X", SURFlista)
print("Salvo")
del SURFlista[:]




# -------------------------------------------------------------
# Testes



for cont in range(0, 300):
    # montamos a imagem
    imagem = cv2.imread("testes\\train_5a_01" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona o descritor na SURFlista
    SURFlista.append(descritor)

print("Salvando descritor Testes surf-Z, tamanho:" + str(len(SURFlista)))
numpy.save("Testes_surf-Z", SURFlista)
print("Salvo")
del SURFlista[:]





for cont in range(0, 300):
    # montamos a imagem
    imagem = cv2.imread("testes\\train_53_01" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona o descritor na SURFlista
    SURFlista.append(descritor)

print("Salvando descritor Testes surf-S, tamanho:" + str(len(SURFlista)))
numpy.save("Testes_surf-S", SURFlista)
print("Salvo")
del SURFlista[:]



for cont in range(0, 300):
    # montamos a imagem
    imagem = cv2.imread("testes\\train_58_01" + "{0:03}".format(cont) + ".png")
    # acha os descritores e os pontos chave
    ponto_chave, descritor = surf.detectAndCompute(imagem, None)
    # adiciona o descritor na SURFlista
    SURFlista.append(descritor)

print("Salvando descritor Testes surf-X, tamanho:" + str(len(SURFlista)))
numpy.save("Testes_surf-X", SURFlista)
print("Salvo")
del SURFlista[:]