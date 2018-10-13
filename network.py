import matplotlib.pyplot as plt
import numpy as np
import math
import random
from os import system
import datetime
np.set_printoptions(precision=3)



def uni(a, b):
	c = random.uniform(0, 1)
	if c < 0.5:
		return random.uniform(a, b)
	return -random.uniform(a, b)

class NeuralNetwork(object):
	def identity(self, x):
		return x
	def d_identity(self, x):
		return 1

	def ReLU(self, x):
		return max(0, x)

	def d_ReLU(self, x):
		if x < 0:
			return 0
		elif x == 0:
			return 0.5
		return 1

	def LReLU(self, x):
		if x < 0:
			return 0.01*x
		return x

	def d_LReLU(self, x):
		if x < 0:
			return 0.01
		elif x == 0:
			return 0.505
		return 1



	def sign(self, x):
		if x < 0:
			return -1
		elif x > 0:
			return 1
		return 0


	def sigmoid(self, x, a=1):
		return (1 + np.exp(-x*a))**(-1)

	def d_sigmoid(self, x, a=1):
		return self.sigmoid(x) * (1 - self.sigmoid(x))

	def sign(self, x):
		if x < 0:
			return -1
		elif x > 0:
			return 1
		return 0

	def diff(self, target, y):
		return (target-y)


	def __init__(self, input_size=1, loss_function="difference", input_bias=[]):
		self.n 				= 1
		self.weights 		= [ [ [] ] ]
		self.values 		= [[1] * input_size]
		self.activation 	= [self.identity]
		self.d_activation 	= [self.d_identity]
		self.input_size 	= input_size
		self.sum_field		= [ [0]*input_size ]
		self.gradient		= [ [0]*input_size ]
		self.lsize			= [input_size]

		self.eta 			= 1

		if len(input_bias) == 0:
			self.bias = [[0]*input_size]
		elif len(input_bias) != input_size:
			raise NameError("Input size and input layer bias size mismatch")
		else:
			self.bias = [input_bias]
		
		if loss_function == "difference":
			self.loss = self.diff
		else:
			raise NameError("Invalid loss function '%s'" % loss_function)

	def set_input(self, values):
		if len(values) != self.input_size:
			raise NameError("Given input size and network's input layer size mismatch")
		for i in range(len(values)):
			self.values[0][i] = values[i] + self.bias[0][i]
			self.sum_field[0][i] = values[i]

	def add_layer(self, layer_size=1, activation="ReLU", weights=[], weights_fill=[1.],
			bias=[]):


		if len(weights) == 0:
			n = layer_size
			m = self.lsize[-1]
			weights = [[1]*self.lsize[-1]] * layer_size
			for i in range(layer_size):
				for j in range(self.lsize[-1]):
					weights[i][j] = weights_fill[(i*n+j)%len(weights_fill)]
		elif len(weights) != layer_size:
			raise NameError("Weights size and layer size mismatch: %d and %d" % (len(weights), layer_size))
		elif len(weights) == layer_size:
			for i in range(len(weights)):
				if len(weights[i]) != self.lsize[-1]:
					raise NameError("Weights size and prev layer size mismatch")

		if activation == "ReLU":
			self.activation.append(self.ReLU)
			self.d_activation.append(self.d_ReLU)
		elif activation == "identity":
			self.activation.append(self.identity)
			self.d_activation.append(self.d_identity)
		elif activation == "LReLU":
			self.activation.append(self.LReLU)
			self.d_activation.append(self.d_LReLU)
		elif activation == "sigmoid":
			self.activation.append(self.sigmoid)
			self.d_activation.append(self.d_sigmoid)
		else:
			raise NameError("Wrong activation function: '%s'" % activation)
		self.n += 1

		self.weights.append(weights)
		self.values.append([1] * layer_size)

		self.sum_field.append([0] * layer_size)
		self.lsize.append(layer_size)
		self.gradient.append([0]*layer_size)
		if len(bias) == layer_size:
			self.bias.append(bias)
		else:
			self.bias.append([0]*layer_size)

	def forward_pass(self, input=[]):
		self.set_input(input)
		for i in range(1, self.n):
			for k in range(self.lsize[i]):
				self.sum_field[i][k] = 0
				for j in range(self.lsize[i-1]):
					self.sum_field[i][k] += self.values[i-1][j] * self.weights[i][k][j] 				
				self.values[i][k] = self.activation[i](self.sum_field[i][k]+self.bias[i][k])

		#print(np.array(self.values[-1]))


	def _gradient_descent(self, errors=[], moment=1):
		if len(errors) != self.lsize[-1]:
			raise NameError("Errors size and output layer size mismatch")

		for i in range(self.lsize[-1]):
			self.gradient[-1][i] = errors[i] * self.d_activation[-1](self.sum_field[-1][i])

		for layer in range(self.n-2, -1, -1):
			for j in range(self.lsize[layer]):
				self.gradient[layer][j] = self.d_activation[layer](self.sum_field[layer][j])

				sum_wdelta = 0
				for k in range(self.lsize[layer+1]):
					sum_wdelta += self.weights[layer+1][k][j] * self.gradient[layer+1][k]
				self.gradient[layer][j] *= sum_wdelta
			#print("Layer %s, gradient: %s" % (layer, self.gradient[layer]))

		for i in range(self.lsize[0]):
			self.bias[0][i] = self.bias[0][i] + self.eta * self.gradient[0][i]

		for layer in range(1, self.n, 1):
			for j in range(self.lsize[layer]):
				for i in range(self.lsize[layer-1]):
					self.weights[layer][j][i] = self.weights[layer][j][i]*moment + \
						+ self.eta * self.gradient[layer][j] * self.values[layer-1][i]

				self.bias[layer][j] = self.bias[layer][j] + self.eta*self.gradient[layer][j]
			#print("Layer %s, weights: %s" % (layer, self.weights[layer]))

	def calculate_error(self, target):
		if len(target) != len(self.values[-1]):
			raise NameError("Target size and output layer size mismatch")
		E = 0
		errors = []
		for i in range(len(self.values[-1])):
			E += self.loss(target[i], self.values[-1][i])**2
			errors.append(self.loss(target[i], self.values[-1][i]))
		E /= len(target)
		return E, errors

	def update_eta(self, oldE=1, newE=1, lambd=0.001):
		delta = newE - (1+lambd)*oldE
		if delta > 0:
			self.eta = self.eta * (1-lambd)
		else:
			self.eta = self.eta * (1+lambd)


	def train(self, dataX=[], dataY=[], epochs=10, eta=1, dynamic_eta=False, dropout=0,
				batch_size=1, moment=1):
		if len(dataX) != len(dataY):
			raise NameError("dataX and dataY size mismatch: %s and %s" % (len(dataX), len(data(Y))))
		oldE = 0
		newE = 0
		self.eta = eta
		for ep in range(epochs):
			if ep % 10 == 0:
				system("clear")
			newE = 0
			cnt = 0
			for j in range(0, len(dataX)-batch_size+1, batch_size):
				errors = [0] * self.lsize[-1]
				for k in range(batch_size):
					self.forward_pass(input=dataX[j+k])
					for i in range(self.lsize[-1]):
						errors[i] += self.loss(dataY[j+k][i], network.values[-1][i])
				for i in range(len(errors)):
					errors[i] /= batch_size

				self._gradient_descent(errors=errors, moment=moment)
				E, a = self.calculate_error(target=dataY[j])
				newE += E
				cnt += 1
				if ep%10 == 0 and j < 5*batch_size:
					print("%s %s" % (ep, j))
					print("Error: %s" % E)
					print("Errors: %s" % errors)
					print("In: %s" % np.array(dataX[j]))
					print("Expected: %s" % np.array(dataY[j]))
					print("Out: %s" % np.array(self.values[-1]))
					print("Eta: %s" % self.eta)
					print()
			newE /= cnt
			self.update_eta(oldE, newE)
			oldE = newE




""" PROBLEMS SPECIFIC CODE """


f = open("dataset", "r")
fl = f.readlines()

dataX = []
dataY = []
mn = 1000000000
mx = -mn
for line in fl:
	x, y = (float(i) for i in line.split(" "))
	dataX.append([x])
	dataY.append([y])
	mn = min(y, mn)
	mx = max(y, mx)
for i in range(len(dataY)):
	dataY[i][0] = (dataY[i][0] - mn) / (mx - mn)


trainX = dataX[:1000]
trainY = dataY[:1000]


testX = dataX[1000:1200]
testY = dataY[1000:1200]




l = 0.01
r = 0.5
network = NeuralNetwork(input_size=1, input_bias=[0])
network.add_layer(2, activation="LReLU", 
		weights=[[uni(l, r)], 
				[uni(l, r)]],
		bias=[1, 1])
network.add_layer(1, activation="LReLU", 
		weights=[[uni(l, r), uni(l, r)]],
		bias=[0])
print("Biases: %s" % np.array(network.bias))


network.train(dataX=trainX, dataY=trainY, epochs=1000, eta=random.uniform(0.01, 0.05), 
				moment=1, batch_size=1)




plt.plot((-5, 5), (0, 0), color="black")
plt.plot((0, 0), (-5, 5,), color="black")
for i in range(100):
	x = testX[i][0]
	y = testY[i][0]
	y = mn + y*(mx-mn)
	plt.scatter(x, y, s=5, c='green')
plt.figure()

plt.plot((-5, 5), (0, 0), color="black")
plt.plot((0, 0), (-5, 5,), color="black")
for i in range(100):
	network.forward_pass(testX[i])
	x = testX[i][0]
	y = testY[i][0]
	y = mn + y*(mx-mn)
	res = network.values[-1][0]
	res = mn + res*(mx-mn)
	plt.scatter(x, res, s=5, c='blue')

plt.show()