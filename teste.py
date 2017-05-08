# coding=utf-8
import numpy
import time
from sklearn.neural_network import MLPClassifier
import pickle
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


MLPClassifier
hidden_layer_sizes=(1000)
activation='logistic' #sigmoid
solver='adam'
alpha=1e-5
batch_size='auto'
learning_rate='adaptative'
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



# gera o model.dat
pickle.dump(rede, open( "model.dat", "wb" ))

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

respostas = numpy.array(respostas)
print("Pronto")





# treinamento




print("Iniciando Treinamento da rede...")

TempoInicio = time.time()


# K FOLD CROSS
k_fold = KFold(n_splits= 5, random_state=None, shuffle=False)
epoca = 1

for idTreino, idTeste in k_fold.split(ArrayCorrigida):
    print " -> rodando epoca: ", epoca

    # seleciona datasets unicos para treinar e testar
    entrada_treino, entrada_teste = ArrayCorrigida[idTreino], ArrayCorrigida[idTeste]
    resposta_treino, resposta_teste = respostas[idTreino], respostas[idTeste]

    print "shape 1: %s  shape 2: %s" % (entrada_treino, resposta_treino)

    print " -> treinando a rede"
    # treina rede
    rede.fit(entrada_treino, resposta_treino)

    print " -> fazendo previsão"
    prediz = rede.predict(entrada_teste)

    print "Erro medio ", rede.score(entrada_treino, resposta_treino)


    # salva info no array de error.txt
    errortxt.append("epoca %i; " % epoca)
    errortxt.append("TREINO: %s | TESTE: %s;" % ((str(len(idTreino)), str(len(idTeste)))))
    errortxt.append("Erro medio treinamento %s" % str(rede.score(entrada_treino, resposta_treino)))

    # atualiza epoca
    epoca = epoca + 1

TempoFim = time.time()

from sklearn.metrics import classification_report, confusion_matrix

print "\n MATRIZ DE CONFUSAO"
print confusion_matrix(resposta_teste, prediz)
print "\n CLASSIFICACAO"
print(classification_report(resposta_teste, prediz))

print("Rede treinada em " + str(TempoFim - TempoInicio) + " segundos")

file = open("error.txt", "w")
for item in errortxt:
  file.write("%s\n" % item)
file.close()



configtxt = (" MLPClassifier \nhidden_layer_sizes : %s, activation : %s, solver : %s, alpha : %s, batch_size : %s,"
                 " learning_rate : %s, learning_rate_init : %s, power_t : %s, max_iter : %s, shuffle : %s, "
                 "random_state : %s, tol : %s, verbose : %s, warm_start : %s, momentum : %s, nesterovs_momentum : %s,"
                  "early_stopping : %s, validation_fraction : %s, beta_1 : %s, beta_2 : %s, epsilon : %s" %

                 (str(hidden_layer_sizes), str(activation), str(solver), str(alpha), str(batch_size),str(learning_rate),
                    str(learning_rate_init), str(power_t), str(max_iter), str(shuffle), str(random_state), str(tol),
                    str(verbose), str(warm_start), str(momentum), str(nesterovs_momentum),str(early_stopping),
                    str(validation_fraction), str(beta_1), str(beta_2), str(epsilon)))

file = open("config.txt", "w")
file.write(configtxt)
file.close()