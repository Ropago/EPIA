# coding=utf-8
import numpy
import time
from sklearn.neural_network import MLPClassifier

HOGS = numpy.load("Treinamento_S.npy")
print("Lendo arquivos de Treinamento S. Tamanho: " + str(len(HOGS)))

HOGX = numpy.load("Treinamento_X.npy")
print("Lendo arquivos de Treinamento X. Tamanho: " + str(len(HOGX)))

HOGZ = numpy.load("Treinamento_Z.npy")
print("Lendo arquivos de Treinamento Z. Tamanho: " + str(len(HOGZ)))

entradas = []
respostas = []

rede = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000), random_state=1,
                     learning_rate='adaptive', max_iter=200)

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
print("Pronto")

print("Iniciando Treinamento da rede...")

TempoInicio = time.time()

rede.fit(ArrayCorrigida, respostas)

TempoFim = time.time()

print("Rede treinada em " + str(TempoFim - TempoInicio) + " segundos")

print("Iniciando leitura e correcao de dimensao de arquivos de teste...")

Teste_S = numpy.load("Testes_S.npy")
Teste_Z = numpy.load("Testes_Z.npy")
Teste_X = numpy.load("Testes_X.npy")

testeSCorrigido = Teste_S.reshape(len(Teste_S), -1)
testeZCorrigido = Teste_Z.reshape(len(Teste_Z), -1)
testeXCorrigido = Teste_X.reshape(len(Teste_X), -1)

print("Analisando Testes:")

print("Letra S:")

RespostasRede = rede.predict(testeSCorrigido)
ContaErros = 0

for resp in RespostasRede:
    if (resp != "S"):
        ContaErros = ContaErros + 1

print("Total de erros: " + str(ContaErros))

print("Letra Z:")

RespostasRede = rede.predict(testeZCorrigido)
ContaErros = 0

for resp in RespostasRede:
    if (resp != "Z"):
        ContaErros = ContaErros + 1

print("Total de erros: " + str(ContaErros))

print("Letra X:")

RespostasRede = rede.predict(testeXCorrigido)
ContaErros = 0

for resp in RespostasRede:
    if (resp != "X"):
        ContaErros = ContaErros + 1

print("Total de erros: " + str(ContaErros))