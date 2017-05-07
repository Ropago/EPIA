# coding=utf-8

# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# http://scikit-learn.org/stable/modules/cross_validation.html#k-fold
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
# http://stackoverflow.com/questions/25889637/how-to-use-k-fold-cross-validation-in-a-neural-network

import numpy
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import time

# carrega arquivo treinamento
treinaZ = numpy.load("Treinamento_surf-Z.npy")
print("Lendo arquivos de Treinamento Z. Tamanho: " + str(len(treinaZ)))

# carrega arquivo treinamento
treinaS = numpy.load("Treinamento_surf-S.npy")
print("Lendo arquivos de Treinamento S. Tamanho: " + str(len(treinaS)))

# carrega arquivo treinamento
treinaX = numpy.load("Treinamento_surf-X.npy")
print("Lendo arquivos de Treinamento X. Tamanho: " + str(len(treinaX)))


# a rede MPL treina com dois arrays : X (samples do treinamento), Y (valores alvo dos samples de trainamento)
entradas = []
respostas = []

for item in treinaZ:
    entradas.append(item)
    respostas.extend("Z")
for item in treinaS:
    entradas.append(item)
    respostas.extend("S")
for item in treinaX:
    entradas.append(item)
    respostas.extend("X")


print("Tamanho da lista de Treinamento: " + str(len(entradas)))
print("Tamanho da lista de respostas: " + str(len(respostas)))

print("Corrigindo dimensao da lista de entradas...")
EntradasArray = numpy.array(entradas)
entradas = EntradasArray.reshape(len(EntradasArray), -1)

respostas = numpy.array(respostas)
respostas = respostas.reshape(len(respostas), -1)
print("Pronto")

print("Iniciando Treinamento da rede...")
TempoInicio = time.time()

# modela a rede
rede = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000), random_state=1,
                     learning_rate='adaptive', max_iter=200)


# cria o kFold Validation com 5 partições. Shuffle ou não?
k_fold = KFold(n_splits= 5, random_state=None, shuffle=False)
for idTreino, idTeste in k_fold.split(entradas):
    # seleciona datasets unicos para treinar e testar
    print("  ------------------------------ INICIO ----------------------")
    print("\n\n TREINO: %s | TESTE: %s" % (idTreino, idTeste))

    # configura os treinamentos
    entrada_treino, entrada_teste = entradas[idTreino], entradas[idTeste]
    resposta_treino, resposta_teste = respostas[idTreino], respostas[idTeste]

    # treina a rede

    print("\n Converte as entradas da rede")
    resposta_treino = numpy.array(resposta_treino)

    print("\n Treina a rede")
    rede.fit(entrada_treino, resposta_treino)



# cuida do tempo
TempoFim = time.time()
print("Rede treinada em " + str(TempoFim - TempoInicio) + " segundos")


