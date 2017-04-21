import random


class Perceptron():
    def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):
        self.amostras = amostras
        self.saidas = saidas
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.limiar = limiar
        self.n_amostras = len(amostras)
        self.n_atributos = len(amostras[0])
        self.pesos = []

    def treinar(self):
        for amostra in self.amostras:
            amostra.insert(0, -1)#Adicionando o valor -1 em cada amostra. Ex: [-1,0,0],[-1,0,1]

        for i in range(self.n_atributos):
            self.pesos.append(random.random())

        self.pesos.insert(0, self.limiar)
        n_epocas = 0 # Contador de épocas
        while True:
            erro = False
            for i in range(self.n_amostras):
                u=0
                for j in range(self.n_atributos+1):# +1 pq adicionou o -1 nos atributos das amostras
                    u += self.pesos[j] * self.amostras[i][j]
                y = self.sinal(u)# Obtem a saida da rede
                # Verifica se a saída da rede é diferente da saida desejada
                if y != self.saidas[i]:
                    erro_aux = self.saidas[i] - y
                    # Ajusta os pesos
                    for j in range(self.n_atributos+1):
                        self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]
                    erro = True # O erro ainda existe

            n_epocas += 1
            # Critério de parada
            if not erro or self.epocas < n_epocas:
                break
        print(n_epocas)

    def teste(self, amostra):
        amostra.insert(0, -1)
        u = 0
        for i in range(self.n_atributos+1):
            u += self.pesos[i] * amostra[i]
        y = self.sinal(u)
        print("saida: %d"% y)

    # Função de ativação
    def degrau(self, u):
        if u >= 0:
            return 1
        return 0

    # Função de ativação
    def sinal(self, u):
        if u >= 0:
            return 1
        return -1



#OR
# amostras = [[0, 0], [0, 1], [1, 0], [1, 1]]
# saidas = [0, 0, 0, 1]
#
# rede = Perceptron(amostras, saidas, taxa_aprendizado=0.1)
# rede.treinar()
# rede.teste([1,1])

amostras = [[0.1, 0.4, 0.7], [0.3, 0.7, 0.2],
            [0.6, 0.9, 0.8], [0.5, 0.7, 0.1]]
saidas = [1, -1, -1, 1]
rede = Perceptron(amostras, saidas)
rede.treinar()
rede.teste([0.5, 0.7, 0.1])