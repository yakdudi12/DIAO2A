class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = [0.0 for _ in range(num_features)]
        self.bias = 0

    def forward(self, x):
        weighted_sum = self.bias
        for i, _ in enumerate(self.weights):
            weighted_sum += x[i] * self.weights[i] # z = wx + b

        if weighted_sum > 0:
            prediction = 1
        else:
            prediction = 0
        return prediction

    def update(self, x, y_true):
        prediction = self.forward(x) #El output del forward, prediction = prediction
        error = y_true - prediction

        self.bias += error
        for i, _ in enumerate(self.weights):
            self.weights[i] += error * x[i]

        return error


#Training Loop
def train(model, X_train, y_train, epochs):
    for i in range(epochs):
        error_c = 0
        for x, y in zip(X_train,y_train):
            error = model.update(x,y)
            error_c += abs(error) # |error|
        print(f"Epoch {i+1} errors {error_c}")

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