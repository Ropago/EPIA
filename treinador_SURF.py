# coding=utf-8

# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# http://scikit-learn.org/stable/modules/cross_validation.html#k-fold
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
import numpy
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler





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

# escala os dados
scaler = StandardScaler()
scaler.fit(entradas)
entradas = scaler.transform(entradas)


# cria o kFold Validation com 5 partições. Shuffle ou não?
k_fold = KFold(n_splits= 5)
for treino, teste in k_fold.split(entradas):
    print("\n\n TREINO: %s | TESTE: %s" % (treino, teste))



# cria o modelo da rede
rede = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000), random_state=1,
                     learning_rate='adaptive', max_iter=20)

# encaixa os dados do treinamento no modelo
# rede.fit(treino, teste)



