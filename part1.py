
#1. Generate the simulated data first using following equation. Sample 120k data as X from uniform distribution [-2*Pi, 2*Pi], 
#then feed the sampled X into the equation to get Y. Randomly select 60K as training and 60 K as testing.

# target function f(x) = 2(2 cos^2(x) − 1)^2 − 1
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(58008)
tf.random.set_seed(58008)

print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))

n = 120_000
X = np.random.uniform(-2*np.pi, 2*np.pi, n).reshape(-1,1)
def f(x):
    return 2 * (2 * np.cos(x)**2 - 1)** 2 - 1

Y = f(X).reshape(-1,1) # reshape(-1,1) turns the 1d array of 120K values into a 2d array with 120k rows and one column each
# -1 means infer the shape of this row (will become 120k)
# 1 means exactly 1 column per feature

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 60000, random_state=58008)

#2. Train 3 versions of Neural Network, with different numbers of hidden layer (NN with 1 hidden layer, 2 hidden layers and 3 hidden layers), 
#using Mean squared error as objective function and error measurement.        

def build_model(n_hidden_layers, n_neurons):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(1,))) # the (1,) means that each input sample has 1 feature, keras expects a tuple

    for _ in range(n_hidden_layers): #just iterate that many times, we dont need an iteration object
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1, activation='linear')) #since we are trying to predict one number

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=.001),
        loss = 'mse',
        metrics = ['mse']
    )
    return model



#3. For each version, try different number of neurals in your NN and replicate the following left plot of Figure 2. 
#(You don’t need to replicate exactly same results below but need to show the performance difference of 3 versions of Neural Networks)

depth_list = [1,2,3]
neurons_list = [5,10,20,50,70,80]
#depth_list = [1]
#neurons_list = [5]
results = []

for depth in depth_list:
    for neurons in neurons_list:
        model = build_model(depth, neurons)
        print(f'building model with {depth} depth and {neurons} neurons')

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_split=.2,
            epochs = 300,
            batch_size = 256,
            verbose=0,
            callbacks=[early_stop]
        )
        
        test_mse = model.evaluate(X_test, y_test, verbose=0)[1] # why do we do [1] here? what is the first index of model evaluate object?
        n_params = model.count_params()

        results.append({
            'depth' : depth,
            'neurons': neurons,
            'params':n_params,
            'test_mse':test_mse
        })

df = pd.DataFrame(results)

plt.figure(figsize=(8,6))

for depth in depth_list:
    sub = df[df['depth'] == depth].sort_values('params')
    plt.plot(
        sub['neurons'],
        sub['test_mse'],
        marker='o',
        label=f"{depth} hidden layers"
    )

plt.xlabel("number of neurons")
plt.ylabel('test mse')
plt.title("deep vs shallow networks")
plt.legend()
plt.grid(True, alpha=.3)
plt.show()