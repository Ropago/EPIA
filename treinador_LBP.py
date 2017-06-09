# coding=utf-8
import numpy
import time
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import KFold

MLPClassifier
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








for ent in HOGZ:
    entradas.append(ent)
    respostas.extend("Z")


print("Pronto")

print("Tamanho da lista de Treinamento: " + str(len(entradas)))
print("Tamanho da lista de respostas: " + str(len(respostas)))

print("Corrigindo dimensao da lista de entradas...")
EntradasArray = numpy.array(entradas, dtype = object)
ArrayCorrigida = EntradasArray.reshape(len(EntradasArray), -1)

respostas = numpy.array(respostas, dtype = object)
print("Pronto")





# treinamento


from sklearn.metrics import accuracy_score

print("Iniciando Treinamento da rede...")

TempoInicio = time.time()


# K FOLD CROSS
k_fold = KFold(n_splits= 5, random_state=None, shuffle=True)
epoca = 0

for idTreino, idTeste in k_fold.split(ArrayCorrigida):
    print (" -> rodando epoca: ", epoca)

    # seleciona datasets unicos para treinar e testar
    entrada_treino, entrada_teste = ArrayCorrigida[idTreino], ArrayCorrigida[idTeste]
    resposta_treino, resposta_teste = respostas[idTreino], respostas[idTeste]

    print ("shape 1: %s  shape 2: %s" % (entrada_treino, resposta_treino))

    print ("Tamanhos:\nEntrada: %s\nRespostas: %s" % (str(len(entrada_treino)), str(len(resposta_treino))))

    print (" -> treinando a rede")
    # treina rede
    rede.fit(entrada_treino, resposta_treino)

    print (" -> fazendo previsão")
    prediz = rede.predict(entrada_teste)

    print (" -> rodando o treinamento para pegar o erro")
    retesta = rede.predict(entrada_treino)

    print ("Erro medio ", 1 - accuracy_score(resposta_teste, prediz))


    # salva info no array de error.txt
    errortxt.append(str(epoca) + ";" + str(1 - accuracy_score(resposta_treino, retesta)) + ";" + str(1 - accuracy_score(resposta_teste, prediz)))

    # atualiza epoca
    epoca = epoca + 1

TempoFim = time.time()

from sklearn.metrics import classification_report, confusion_matrix

print ("\n MATRIZ DE CONFUSAO")
print (confusion_matrix(resposta_teste, prediz))
print ("\n CLASSIFICACAO")
print(classification_report(resposta_teste, prediz))

print("Rede treinada em " + str(TempoFim - TempoInicio) + " segundos")

file = open("error.txt", "w")
for item in errortxt:
  file.write("%s\n" % item)
file.close()



