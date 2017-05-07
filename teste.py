# coding=utf-8
import numpy
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold


HOGS = numpy.load("Treinamento_S.npy")
print("Lendo arquivos de Treinamento S. Tamanho: " + str(len(HOGS)))

HOGX = numpy.load("Treinamento_X.npy")
print("Lendo arquivos de Treinamento X. Tamanho: " + str(len(HOGX)))

HOGZ = numpy.load("Treinamento_Z.npy")
print("Lendo arquivos de Treinamento Z. Tamanho: " + str(len(HOGZ)))

entradas = []
respostas = []
errortxt = []



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
respostas = numpy.array(respostas)
ArrayCorrigida = EntradasArray.reshape(len(EntradasArray), -1)
print("Pronto")





# treinamento




print("Iniciando Treinamento da rede...")

TempoInicio = time.time()


# K FOLD CROSS
k_fold = KFold(n_splits= 5, random_state=None, shuffle=False)
epoca = 1
for idTreino, idTeste in k_fold.split(ArrayCorrigida):
    # seleciona datasets unicos para treinar e testar

    print("\n\n  ÉPOCA ", epoca)
    print("TREINO: %s | TESTE: %s" % ((str(len(idTreino)), str(len(idTeste)))))


    # configura os treinamentos
    entrada_treino, entrada_teste = ArrayCorrigida[idTreino], ArrayCorrigida[idTeste]
    resposta_treino, resposta_teste = respostas[idTreino], respostas[idTeste]

    print "........>"
    # treina rede
    rede.fit(entrada_treino, resposta_treino)

    prediz = rede.predict(entrada_teste)

    print "Erro medio ", rede.score(entrada_treino, resposta_treino)
    from sklearn.metrics import classification_report, confusion_matrix
    print "\n MATRIZ DE CONFUSAO"
    print confusion_matrix(resposta_teste, prediz)
    print "\n CLASSIFICACAO"
    print(classification_report(resposta_teste, prediz))

    epoca = epoca + 1
    errortxt.append("epoca %i; " % epoca)
    errortxt.append("TREINO: %s | TESTE: %s;" % ((str(len(idTreino)), str(len(idTeste)))))
    errortxt.append("Erro medio treinamento %s" % str(rede.score(entrada_treino, resposta_treino)))

TempoFim = time.time()

print("Rede treinada em " + str(TempoFim - TempoInicio) + " segundos")

file = open("error.txt", "w")
file.write(errortxt)
file.close()
