from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


dataset = SupervisedDataSet(2,1)

# Adiciona a tabela XOR
dataset.addSample([0,0], [0])
dataset.addSample([0,1], [1])
dataset.addSample([1,0], [1])
dataset.addSample([1,1], [0])

# dimensões de entrada e saida, argumento 2 é a quantidade de camadas intermediárias
network = buildNetwork(dataset.indim, 4, dataset.outdim, bias=True)

trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.99)

trainer.trainEpochs(1000)

test_data = SupervisedDataSet(2,1)

test_data.addSample([0,0], [0])
test_data.addSample([0,1], [1])
test_data.addSample([1,0], [1])
test_data.addSample([1,1], [0])

trainer.testOnData(test_data, verbose=True)