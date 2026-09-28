
import numpy as np
class MLP:
    def __init__(self, input_features, hidden_size, output_size, lr=0.01, acfunc=None):
        self.weights_input_hidden = np.random.randn(input_features, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.lr = lr
        self.acfunc = acfunc

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)


    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden #producto punto
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def backward(self, x, y, output, lr):
        x = x.reshape(1, -1)
        '''Descenso del gradiente'''
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)
        self.weights_hidden_output -= lr * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= lr * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= lr * np.dot(x.T, hidden_error)
        self.bias_hidden -= lr * np.sum(hidden_error, axis=0, keepdims=True)

#Gemini Pro
class MLP2:
    def __init__(self, input_features, hidden1_size, hidden2_size, output_size, lr=0.01, acfunc="relu"):
        self.acfunc = acfunc

        self.w1 = np.random.randn(input_features, hidden1_size) * 0.1
        self.b1 = np.zeros((1, hidden1_size))
        
        self.w2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
        self.b2 = np.zeros((1, hidden2_size))
        
        self.w3 = np.random.randn(hidden2_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))
        
        self.lr = lr

    def sigmoid(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, x):
        if self.acfunc == "sigmoid":
            self.z1 = np.dot(x, self.w1) + self.b1
            self.a1 = self.sigmoid(self.z1)
                        
            # h2
            self.z2 = np.dot(self.a1, self.w2) + self.b2
            self.a2 = self.sigmoid(self.z2)
                        
            # Out
            self.z3 = np.dot(self.a2, self.w3) + self.b3
            self.a3 = self.softmax(self.z3)

        elif self.acfunc == "relu":
            # h1
            self.z1 = np.dot(x, self.w1) + self.b1
            self.a1 = self.relu(self.z1)
            
            # h2
            self.z2 = np.dot(self.a1, self.w2) + self.b2
            self.a2 = self.relu(self.z2)
            
            # Out
            self.z3 = np.dot(self.a2, self.w3) + self.b3
            self.a3 = self.softmax(self.z3)
            
        return self.a3
    
    def backward(self, x, y, output, lr):
        x = x.reshape(1, -1)

        if self.acfunc == "sigmoid":
            # 1. Error en la salida (Cross-Entropy + Softmax)
            dz3 = output - y
            
            # 2. Error propagado a la Oculta 2 (dz2)
            da2 = np.dot(dz3, self.w3.T)
            dz2 = da2 * self.a2 * (1 - self.a2) # Derivada Sigmoide
            
            # 3. Error propagado a la Oculta 1 (dz1)
            da1 = np.dot(dz2, self.w2.T)
            dz1 = da1 * self.a1 * (1 - self.a1) # Derivada Sigmoide
            
            # 4. Actualización de Pesos y Sesgos
            self.w3 -= lr * np.dot(self.a2.T, dz3)
            self.b3 -= lr * np.sum(dz3, axis=0, keepdims=True)
            
            self.w2 -= lr * np.dot(self.a1.T, dz2)
            self.b2 -= lr * np.sum(dz2, axis=0, keepdims=True)
            
            self.w1 -= lr * np.dot(x.T, dz1)
            self.b1 -= lr * np.sum(dz1, axis=0, keepdims=True)
        elif self.acfunc == "relu":
            # 1. Error en la salida
            dz3 = output - y
            
            # 2. Error propagado a la Oculta 2 (Usando derivada ReLU de z2)
            da2 = np.dot(dz3, self.w3.T)
            dz2 = da2 * self.relu_derivative(self.z2)
            
            # 3. Error propagado a la Oculta 1 (Usando derivada ReLU de z1)
            da1 = np.dot(dz2, self.w2.T)
            dz1 = da1 * self.relu_derivative(self.z1)
            
            # 4. Actualización
            self.w3 -= lr * np.dot(self.a2.T, dz3)
            self.b3 -= lr * np.sum(dz3, axis=0, keepdims=True)
            
            self.w2 -= lr * np.dot(self.a1.T, dz2)
            self.b2 -= lr * np.sum(dz2, axis=0, keepdims=True)
            
            self.w1 -= lr * np.dot(x.T, dz1)
            self.b1 -= lr * np.sum(dz1, axis=0, keepdims=True)
    
#Training Loop
def train_mlp(model, X_train, y_train, epochs):
    hist_dic = {"epoch": [], "loss": []}
    for i in range(epochs):
        total_loss = 0.0
        for x, y in zip(X_train,y_train):
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            output = model.forward(x)
            loss = -np.sum(y * np.log(output + 1e-15))
            total_loss += loss 
            model.backward(x, y, output, model.lr)
        epoch_loss = total_loss / len(y_train)

        hist_dic["epoch"].append(i + 1)
        hist_dic["loss"].append(epoch_loss)
        print(f"Epoch: {i+1} , Loss: {epoch_loss}")

    return hist_dic

# Metricas
def compute_accuracy_mlp(model, x_train, y_train):
    correct = 0.0
    for x, y in zip(x_train,y_train):
        prediction = np.argmax(model.forward(x))
        true_class = np.argmax(y)
        correct += int(prediction == true_class)

    return correct / len(y_train)


#Gemini pro.
def compute_metrics_mlp(model, x_data, y_data_ohe,num_classes=10):
    # Vectores para guardar TP, FP y FN por cada clase (del 0 al 9)
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for x, y in zip(x_data, y_data_ohe):
        x = x.reshape(1, -1)
        prediction = np.argmax(model.forward(x))
        true_class = np.argmax(y)
        
        if prediction == true_class:
            tp[true_class] += 1
        else:
            fp[prediction] += 1
            fn[true_class] += 1

    precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    
    f1_per_class = np.divide(2 * precision_per_class * recall_per_class, 
                             precision_per_class + recall_per_class, 
                             out=np.zeros_like(precision_per_class), 
                             where=(precision_per_class + recall_per_class) != 0)

    # El MACRO AVERAGE es el promedio simple de las métricas de las 10 clases
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    return macro_precision, macro_recall, macro_f1