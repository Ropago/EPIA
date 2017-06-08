# coding=utf-8

import numpy
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import time
import pickle


# carrega arquivo treinamento da letra Z
treinaZ = numpy.load("Treinamento_surf-Z.npy")
print("Lendo arquivos de Treinamento Z. Tamanho: " + str(len(treinaZ)))

# carrega arquivo treinamento da letra S
treinaS = numpy.load("Treinamento_surf-S.npy")
print("Lendo arquivos de Treinamento S. Tamanho: " + str(len(treinaS)))

# carrega arquivo treinamento da letra X
treinaX = numpy.load("Treinamento_surf-X.npy")
print("Lendo arquivos de Treinamento X. Tamanho: " + str(len(treinaX)))


# distribui o que carregou em dois arrays
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


# configura a rede MLP

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
# inicializa a rede
rede = MLPClassifier(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t,
                     max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum,
                     early_stopping, validation_fraction, beta_1, beta_2, epsilon)

# gera o model.dat
pickle.dump(rede, open( "model.dat", "wb" ))

# adiciona ao config.txt
configtxt = ("\n\nMLPClassifier (SURF) \nhidden_layer_sizes : %s \nactivation : %s \nsolver : %s \nalpha : %s \nbatch_size : %s \n"
                 " learning_rate : %s \nlearning_rate_init : %s \npower_t : %s \nmax_iter : %s \nshuffle : %s \n"
                 "random_state : %s \ntol : %s \nverbose : %s \nwarm_start : %s \nmomentum : %s \nnesterovs_momentum : %s \n"
                  "early_stopping : %s \nvalidation_fraction : %s \nbeta_1 : %s \nbeta_2 : %s \nepsilon : %s" %
                 (str(hidden_layer_sizes), str(activation), str(solver), str(alpha), str(batch_size),str(learning_rate),
                    str(learning_rate_init), str(power_t), str(max_iter), str(shuffle), str(random_state), str(tol),
                    str(verbose), str(warm_start), str(momentum), str(nesterovs_momentum),str(early_stopping),
                    str(validation_fraction), str(beta_1), str(beta_2), str(epsilon)))
with open("config.txt", "a") as myfile:
    myfile.write(configtxt)
myfile.close()

# Corrigindo dimensao da lista de entradas
print("Corrigindo dimensao da lista de entradas...")
entradas = numpy.array(entradas)
entradas = entradas.reshape(len(entradas), -1)
respostas = numpy.array(respostas)

# marca inicio do k-fold
print("Iniciando Treinamento da rede...")
TempoInicio = time.time()

# cria o kFold Validation com 5 partições.
k_fold = KFold(n_splits= 5, random_state=None, shuffle=True)
for idTreino, idTeste in k_fold.split(entradas):
    # seleciona datasets unicos para treinar e testar
    print("  ------------------------------ INICIO ----------------------")
    print("\n\n TREINO: %s | TESTE: %s" % (idTreino, idTeste))

    # configura os treinamentos
    entrada_treino, entrada_teste = entradas[idTreino], entradas[idTeste]
    resposta_treino, resposta_teste = respostas[idTreino], respostas[idTeste]

    print "shape 1: %s  shape 2: %s" % (entrada_treino.shape, resposta_treino.shape)

    # treina a rede
    print("\n Treina a rede")
    rede.fit(entrada_treino, resposta_treino)

    # faz a previsão
    print " -> fazendo previsão"
    prediz = rede.predict(entrada_teste)


# cuida do tempo
TempoFim = time.time()
print("Rede treinada em " + str(TempoFim - TempoInicio) + " segundos")


