from sklearn.neural_network import MLPClassifier
import cv2


X = [[0., 0.], [1., 1.], [2., 2.]]
y = [0, 1, 2]
rede = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1)

rede.fit(X, y)

resp = rede.predict([[2., 2.], [-1., -2.], [0., 1.], [2., 1.]])

print(resp)

hog = cv2.HOGDescriptor()

im = cv2.imread("treinamento\\train_5a_00000.png")

h = hog.compute(im)

print(h)