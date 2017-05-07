import numpy
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
                     learning_rate='adaptive', max_iter=20)

#X_Corrigido =

for ent in HOGS:
    entradas.append(ent)
    respostas.extend("S")

for ent in HOGX:
    entradas.append(ent)
    respostas.extend("X")

for ent in HOGZ:
    entradas.append(ent)
    respostas.extend("Z")


print("Começa Rede")

print(len(HOGS))
print(len(respostas))

novoArray = numpy.array(entradas)

#print(entradas[1])
entradaNova = novoArray.reshape(len(novoArray), -1)

rede.fit(entradaNova, respostas)

Teste_S = numpy.load("Testes_S.npy")

testesCorrigidos = Teste_S.reshape(len(Teste_S), -1)

resp = rede.predict(testesCorrigidos)
print(resp)