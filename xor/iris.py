from pybrain.supervised.trainers import BackpropTrainer
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork


# iris-setosa = 0
# Iris-versicolor = 1
# Iris-virginica = 2

entradas = np.genfromtxt('iris.data', delimiter=',', usecols=(0,1,2,3))
saidas = np.genfromtxt('iris.data', delimiter=',', usecols=(4))


# Pegando 35 de cada para treino e deixando 15 de cada para teste

entradas_treino = np.concatenate((entradas[:35], entradas[50:85], entradas[100:135]))
saidas_treino = np.concatenate((saidas[:35], saidas[50:85], saidas[100:135]))

entradas_teste = np.concatenate((entradas[35:50], entradas[85:100], entradas[135:150]))
saidas_teste = np.concatenate((saidas[35:50], saidas[85:100], saidas[135:150]))

treinamento = SupervisedDataSet(4,1)
for i in range(len(entradas_treino)):
    treinamento.addSample(entradas_treino[i], saidas_treino[i])
print(len(treinamento))
print(treinamento.indim)
print(treinamento.outdim)

# Criando a rede:
rede = buildNetwork(treinamento.indim, 2, treinamento.outdim, bias=True)
trainer = BackpropTrainer(rede, treinamento, learningrate=0.01, momentum=0.3)

for epoca in range(1000):
    trainer.train()


# Criando o teste da rede:

teste = SupervisedDataSet(4,1)
for i in range(len(entradas_teste)):
    teste.addSample(entradas_teste[i], saidas_teste[i])

trainer.testOnData(teste, verbose=True)