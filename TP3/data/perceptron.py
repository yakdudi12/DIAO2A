import numpy as np
class PerceptronLineal:
    def __init__(self, num_features, lr=0.01):
        self.num_features = num_features
        self.weights = [0.0 for _ in range(num_features)]
        self.bias = 0
        self.lr = lr

    def forward(self, x):
        weighted_sum = self.bias
        for i, _ in enumerate(self.weights):
            weighted_sum += x[i] * self.weights[i] # z = wx + b

        if weighted_sum > 0:
            prediction = 1
        else:
            prediction = 0
        return prediction

    def net_imput(self,x):
        z = self.bias
        for i in range(len(self.weights)):
            z += x[i] * self.weights[i]
        return z

    def update(self, x, y_true):
        z = self.net_imput(x)
        error = y_true - z #true label menos el net imput
        self.bias += self.lr * error   #learning rate * error
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * x[i]  #learning rate * los pesos

        return error ** 2

import numpy as np
class PerceptronNoLineal:
    '''Activation Sigmoid, loss BCE (Binary Cross-Entropy)'''
    def __init__(self, num_features, lr=0.01):
        self.num_features = num_features
        self.weights = [0.0 for _ in range(num_features)]
        self.bias = 0
        self.lr = lr

    def forward(self, x):
        weighted_sum = self.bias
        for i, _ in enumerate(self.weights):
            weighted_sum += x[i] * self.weights[i] # z = wx + b

        if weighted_sum > 0:
            prediction = 1
        else:
            prediction = 0
        return prediction

    def net_imput(self,x):
        z = self.bias
        for i in range(len(self.weights)):
            z += x[i] * self.weights[i]
        return z

    def predic_prob(self, x):
        z = self.net_imput(x)
        z_bounded = max(-500.0, min(500.0, z)) 
        return 1.0 / (1.0 + np.exp(-z_bounded))

    def update(self, x, y_true):
        prob = self.predic_prob(x)
        error = y_true - prob #true label menos la probabilidad

        self.bias += self.lr * error   #learning rate * error
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * x[i]  #learning rate * los pesos

        eps = 1e-15
        p_safe = max(eps,min(1-eps, prob)) #para controlar los gradientes
        bce_loss = -(y_true * np.log(p_safe) + (1 - y_true) * np.log(1 - p_safe))
        #-(yi * log(^yi) + 1 - yi * log(1- ^yi))

        return bce_loss

    
#Training Loop
def train(model, X_train, y_train, epochs):
    hist_dic = {"epoch": [], "loss": []}
    for i in range(epochs):
        bce_total = 0.0
        for x, y in zip(X_train,y_train):
            loss = model.update(x,y)
            bce_total += loss 
        epoch_bc = bce_total / len(y_train)

        hist_dic["epoch"].append(i + 1)
        hist_dic["loss"].append(epoch_bc)
        print(f"Epoch: {i+1} , Loss: {epoch_bc} ")

    return hist_dic

# Metricas
def compute_accuracy(model, x_train, y_train):
    correct = 0.0
    for x, y in zip(x_train,y_train):
        prediction = model.forward(x)
        correct += int(prediction==y)

    return correct / len(y_train)


#Gemini pro.
def compute_metrics(model, x_data, y_data):
    tp = 0.0  
    fp = 0.0  
    fn = 0.0  
    for x, y in zip(x_data, y_data):
        prediction = model.forward(x)
        
        if prediction == 1 and y == 1:
            tp += 1
        elif prediction == 1 and y == 0:
            fp += 1
        elif prediction == 0 and y == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score

def plot_boundary(model,xtrain,ytrain):
    w1, w2 = model.weights[0], model.weights[1]
    b = model.bias

    x1_min = xtrain[:, 0].min()
    x1_max = xtrain[:, 0].max()
    x2_min = (-(w1 * x1_min) - b) / w2
    x2_max = (-(w1 * x1_max) - b) / w2
    
    return x1_min, x1_max, x2_min, x2_max