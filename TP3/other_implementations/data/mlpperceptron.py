
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
    def __init__(self, input_features, hidden1_size, hidden2_size, output_size, lr=0.01, acfunc="relu",
                 optimizer="sgd", beta1=0.9, beta2=0.999, eps=1e-8):
        if optimizer not in ("sgd", "momentum", "rmsprop", "adam"):
            raise ValueError(f"Optimizador desconocido: {optimizer}")
        self.acfunc = acfunc
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.w1 = np.random.randn(input_features, hidden1_size) * 0.1
        self.b1 = np.zeros((1, hidden1_size))
        
        self.w2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
        self.b2 = np.zeros((1, hidden2_size))
        
        self.w3 = np.random.randn(hidden2_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))
        
        self.lr = lr

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def _step(self, grads, lr):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.optimizer == "sgd":
                p -= lr * g
            elif self.optimizer == "momentum":
                self.m[i] = self.beta1 * self.m[i] + g
                p -= lr * self.m[i]
            elif self.optimizer == "rmsprop":
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
                p -= lr * g / (np.sqrt(self.v[i]) + self.eps)
            elif self.optimizer == "adam":
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

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
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        if self.acfunc == "sigmoid":
            # 1. Error en la salida (Cross-Entropy + Softmax)
            dz3 = output - y
            
            # 2. Error propagado a la Oculta 2 (dz2)
            da2 = np.dot(dz3, self.w3.T)
            dz2 = da2 * self.a2 * (1 - self.a2) # Derivada Sigmoide
            
            # 3. Error propagado a la Oculta 1 (dz1)
            da1 = np.dot(dz2, self.w2.T)
            dz1 = da1 * self.a1 * (1 - self.a1) # Derivada Sigmoide
        elif self.acfunc == "relu":
            # 1. Error en la salida
            dz3 = output - y
            
            # 2. Error propagado a la Oculta 2 (Usando derivada ReLU de z2)
            da2 = np.dot(dz3, self.w3.T)
            dz2 = da2 * self.relu_derivative(self.z2)
            
            # 3. Error propagado a la Oculta 1 (Usando derivada ReLU de z1)
            da1 = np.dot(dz2, self.w2.T)
            dz1 = da1 * self.relu_derivative(self.z1)

        grads = [
            np.dot(x.T, dz1), np.sum(dz1, axis=0, keepdims=True),
            np.dot(self.a1.T, dz2), np.sum(dz2, axis=0, keepdims=True),
            np.dot(self.a2.T, dz3), np.sum(dz3, axis=0, keepdims=True),
        ]
        # Promedio sobre el lote (con una sola muestra no cambia nada)
        grads = [g / batch_size for g in grads]
        self._step(grads, lr)

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

def evaluate_mlp(model, X, y_ohe):
    '''Loss y accuracy sobre todo el conjunto en una sola pasada matricial'''
    output = model.forward(X)
    loss = -np.sum(y_ohe * np.log(output + 1e-15)) / len(y_ohe)
    acc = np.mean(output.argmax(axis=1) == y_ohe.argmax(axis=1))
    return loss, acc

#Training Loop por mini-batches, con shuffle por epoch y seguimiento de validacion
def train_mlp_minibatch(model, X_train, y_train, epochs, batch_size=32, X_val=None, y_val=None,
                        seed=None, restore_best=True):
    rng = np.random.default_rng(seed)
    n = len(y_train)
    hist_dic = {"epoch": [], "loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_params = -1.0, 0, None

    for i in range(epochs):
        idx = rng.permutation(n)
        total_loss = 0.0
        correct = 0
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            x, y = X_train[batch], y_train[batch]
            output = model.forward(x)
            total_loss += -np.sum(y * np.log(output + 1e-15))
            correct += np.sum(output.argmax(axis=1) == y.argmax(axis=1))
            model.backward(x, y, output, model.lr)

        hist_dic["epoch"].append(i + 1)
        hist_dic["loss"].append(total_loss / n)
        hist_dic["train_acc"].append(correct / n)
        msg = f"Epoch: {i+1} , Loss: {total_loss / n:.5f} , Acc: {correct / n:.4f}"

        if X_val is not None:
            val_loss, val_acc = evaluate_mlp(model, X_val, y_val)
            msg += f" , Val loss: {val_loss:.5f} , Val acc: {val_acc:.4f}"
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, i + 1
                best_params = [p.copy() for p in model.params]
        else:
            val_loss, val_acc = np.nan, np.nan
        hist_dic["val_loss"].append(val_loss)
        hist_dic["val_acc"].append(val_acc)
        print(msg)

    if restore_best and best_params is not None:
        for p, best in zip(model.params, best_params):
            p[...] = best
        print(f"Se restauran los pesos de la epoch {best_epoch} (val acc: {best_val_acc:.4f})")

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