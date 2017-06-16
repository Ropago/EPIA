import sys
import pickle
import codecs
from os.path import join

def deserialise3(name):
    f = open(name, 'rb')
    p = pickle.Unpickler(f)
    obj = p.load()
    f.close()
    return obj

def deserialise2(name):
  file = open(name, "rb")
  data = pickle.load(file)
  file.close()
  return data


def main():

  script, filename = sys.argv
  
  if(sys.version_info[0] == 2):
    deserialise = deserialise2
  else:
    deserialise = deserialise3
    
  print('Arquivo com o modelo serializado: {0}'.format(filename))

  (w0, w1) = deserialise(filename)
  (w0nr, w0nc) = w0.shape
  (w1nr, w1nc) = w1.shape

  print('Pesos da camada escondida: {0} x {1}'.format(w0nr, w0nc))
  print('Pesos da camada de saida : {0} x {1}'.format(w1nr, w1nc))
  print("")


if __name__ == '__main__':
    main()
