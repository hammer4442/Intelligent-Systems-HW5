import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math


# MATH SYMBOL VARIABLES:

# W1: weights for the hidden layer, stored in INPUT_LENGTHxHIDDEN_LAYER_NEURONS numpy array
# W2: weights for the output layer, stored in HIDDEN_LAYER_NEURONSxOUTPUT_LAYER_NEURONS numpy array
# s: a sum of input values, generic if not followed by j or i
# si: sum of products of weights and inputs for hidden layer, stored as a float
# sj: sum of products of weights and inputs for output layer, stored as a float
# x: input vetor, stored as INPUT_LENGTH element numpy array
# X: input matrix, stores all input vectors for a given data set, stored in a SUBSET_SIZExINPUT_LENGTH numpy array
# h: output vector of the hidden layer neuron, stored in a HIDDEN_LAYER_NEURONS element numpy array 
# y_hat: output of the output layer neurons, stored in a OUTPUT_LAYER_NEURONS element numpy array 





# MODEL VARIABLES

# Input length 
INPUT_LENGTH = 784
# Number of neurons in the hidden layer
HIDDEN_LAYER_NEURONS=144
# Number of Neurons in the output layer
OUTPUT_LAYER_NEURONS=10
# Number of epochs ran
EPOCHS=200
# Learning Rate
LEARNING_RATE=0.02
# Momentum Constant 
MOMENTUM=0.00
# Number of images from the trainng set looked at per epoch
SUBSET_SIZE=800
# Size of a side of the SOFM
GRID_SIZE=12

# ACTIVATION FUNCTION AND ITS DERIVITIVE

# Activation Function 
# Sigmoid 
def f(s):
    return 1/(1+np.exp(-1*s))


# Derivitive of Activation Function
def df(s):
    t=np.exp(-1*s)
    return t/((1+t)**2)



# INITIALIZING DATA AND WEIGHTS
# Reads data from data and label files and returns contents in forms of numpy arrays 
# Returns two arrays:
# Data Matrix: stored as a SUBSET_SIZExINPUT_LENGTH matrix representing the input values with the bias added 
# Label Vector: stored as a SUBSET_SIZE element numpy array with each element being the label of 
# it's corresponding index's digit in the Data Matrix 
def load_dataset(data_file, label_file):
    data = np.loadtxt(data_file)
    labels = np.loadtxt(label_file, dtype=int)
    # Insert bias = 1.0 as first input
    return data.astype(np.float32), labels


# Loads previous weights from encoder
def init_weights_used(n_input,n_hidden, n_output):
    with np.load("q2weights.npz") as data:
        W1 = data['W1']
    # feautremap_super(W1, runname)
    rng = np.random.default_rng(0)
    std_dev2= (6/(n_hidden+n_output))**.5
    W2 = rng.normal(0, std_dev2, size=(n_output, n_hidden)).astype(np.float32)
    return W1,W2

# Creates four files, two pairs of two 
# The first is a training data file with 4000 radnomly selected items, 400 for each digit
# The second is a training label file with the corrspodning label for the digit in the data file
# in the same positon in the label file 
# The thrid is a test data file, it takes the remaining 100 images for each digit
# The furth is a test label file with the labels corresponding to the iamge in the same postion in the test data file
def get_data(runname):

    # Opening files
    digits=open("MNISTnumImages5000_balanced.txt", "r")
    dlines=digits.readlines()
    labels=open("MNISTnumLabels5000_balanced.txt", "r")
    llines=labels.readlines()

    train = open(f"run_{runname}_results/TrainingData{runname}.txt", "w")
    r_labels = open(f"run_{runname}_results/TrainingLabels{runname}.txt", "w")

    test = open(f"run_{runname}_results/TestData{runname}.txt", "w")
    t_labels = open(f"run_{runname}_results/TestLabels{runname}.txt", "w")

    # Seeder represents a list on indexes to be randomized
    seeder=[]
    for x in range(10):
        for y in range(400):
            seeder.append(y+(x*500))
    random.shuffle(seeder)

    # Filling the training data files with the randomized indexes
    for z in seeder:
        train.write(dlines[z])
        r_labels.write(llines[z])
    
    train.close()
    r_labels.close()

    #  Filling the testing data 
    for a in range(10):
        for b in range(100):
            test.write(dlines[b+400+(a*500)])
            t_labels.write(llines[b+400+(a*500)])
    
    test.close()
    t_labels.close()
    digits.close()
    labels.close()
    return []

# FUNCTIONS USED IN TRAINING

# Applies input vector, x, to the wieghts and returns 4 elements in output
def apply_weights(W1, W2, x):
    si = W1 @ x
    winner=np.argmax(si)
    winner_h=np.zeros(len(si))
    winner_h[winner]=1.0
    sj = W2 @ winner_h
    y_hat = f(sj)
    return si, winner_h, sj, y_hat

# Backpopagates the output from a single image and returns new weights and the changes that occured
# change_weights does not change W1
def change_weights(W1, W2, x, label, prev_dW2):
    # target = one-hot
    target = np.zeros(10, dtype=np.float32)
    target[label] = 1.0

    # Get the output for this image
    si, h, sj, y_hat = apply_weights(W1, W2, x)

    # Output layer delta, stored in OUTPUT_LAYER_NEURONS element numpy array
    delta_W2 = -(target-y_hat) * df(sj)

    # Output layer gradient, stored in HIDDEN_LAYER_NEURONSxOUTPUT_LAYER_NEURONS numpy array
    dW2 = np.outer(delta_W2, h)
    
    # Need these saved so it an be used for momentum in the future
    # Changes to W2, stored in HIDDEN_LAYER_NEURONSxOUTPUT_LAYER_NEURONS numpy array
    upd_W2 = -LEARNING_RATE * dW2 + MOMENTUM * prev_dW2

    # Applying the changes to the weights 
    # Adding arrays of identical sizes
    W2_new = W2 + upd_W2
    

    return W2_new, upd_W2



# Trians the model over a course of epochs equal to EPOCHS
# When the final epoch is run the final weights and two lists of the erros for every 10 epochs is returned
def train_just_output(W1, W2, X_train, y_train, X_test, y_test):
    # These arrays will hold the previous changes to the weights they represent
    # Used in momentum calculation in backpropagation
    prev_dW2 = np.zeros_like(W2)

    # Arrays to hold errors for the training and test set respectivly
    # Will get appeneded at the same time so they should always have the same size 
    train_errors = []
    test_errors = []

    # Get the intial error and add them to the arrays
    train_errors.append(error_fraction(W1, W2, X_train, y_train))
    test_errors.append(error_fraction(W1, W2, X_test, y_test))
    print(f"Initial train error: {train_errors[-1]:.4f}, test error: {test_errors[-1]:.4f}")

    # Runs for a set number of epochs
    # Could easily be changed to a while loop looking at the error 
    for epoch in range(1, EPOCHS + 1):

        # Randomly choose SUBSET_SIZE elements from the training set 
        seed = np.random.choice(len(X_train), size=SUBSET_SIZE, replace=False)
        for i in seed:
            x = X_train[i]
            y = y_train[i]
            W2, prev_dW2 = change_weights(W1, W2, x, y,prev_dW2)

        # Record the error for traing and test
        if epoch % 10 == 0:
            tr = error_fraction(W1, W2, X_train, y_train)
            te = error_fraction(W1, W2, X_test, y_test)
            train_errors.append(tr)
            test_errors.append(te)
            print(f"Epoch {epoch}: train error={tr:.4f}, test error={te:.4f}")

    # Error arrays are size EPOCHS/10 + 1
    return W1, W2, train_errors, test_errors


# Returns error fraction
def error_fraction(W1, W2, X, y):
    wrong = 0
    for i in range(len(X)):
        y_hat = apply_weights(W1, W2, X[i])[3]
        pred = np.argmax(y_hat)
        if pred != y[i]:
            wrong += 1
    return wrong / len(X)



# PLOTTING FUNCTIONS

# Plot the errors vs epochs 
# Saves the plots to the file for the run
def plot_errors(errors1, errors2, case):
    plt.figure(figsize=(7, 5))
    x_vals=np.arange(len(errors1))*10
    plt.plot(x_vals, errors1, label="Training Set")
    plt.plot(x_vals, errors2, label="Test Set")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Error vs Epochs")
    plt.legend()
    plt.savefig(f"run_{runname}_results/case{case}/Errors{case}.png")
    plt.close()

# Plots confusion matrix for data set X
def make_confusion(W1, W2, X, y, title,case):
    preds = []
    for i in range(len(X)):
        y_hat= apply_weights(W1, W2, X[i])[3]
        preds.append(np.argmax(y_hat))
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.savefig(f"run_{runname}_results/case{case}/{title}_Confusion_Matrix{case}.png")
    plt.close()
    return cm

# Finds the error and error standard deviation for each digit across the run on the given set
def per_digit_error(W1, W2, X,y):
    size=int(len(X)/OUTPUT_LAYER_NEURONS)
    j=np.zeros(OUTPUT_LAYER_NEURONS, dtype=np.float32)
    std_devs=np.zeros(OUTPUT_LAYER_NEURONS, dtype=np.float32)
    for k in range(len(j)):
        j[k]+=error_fraction(W1,W2,X[(k*size):(k*size+100)],y[(k*size):(k*size+100)])
    j=j*.01
    for x in range(len(j)):
        std_devs[x]+=(error_fraction(W1,W2,X[(x*size):(x*size+100)],y[(x*size):(x*size+100)])-j[x])**2
    std_devs=std_devs*.01
    std_devs=np.sqrt(std_devs)
    return j, std_devs

# Plots error and error standard deviation for each digit across the run on given set data_set
def plot_digit_info(vals, label, runname, data_set, case):
    labels=['0','1','2','3','4','5','6','7','8','9']
    x_vals=np.arange(10)
    plt.figure(figsize=(7, 5))
    plt.bar(x_vals, vals)
    plt.xticks([y for y in range(10)],labels)
    plt.xlabel("Digit")
    plt.ylabel(label)
    plt.title(f"Per Digit {label} for {data_set} Set for Case {case}")
    plt.savefig(f"run_{runname}_results/case{case}/{runname}_{label}_{data_set}{case}.png", bbox_inches='tight', dpi=100)
    plt.close()

def convert_to_square(arr):
    ret=np.zeros((GRID_SIZE,GRID_SIZE))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ret[r][c]=arr[(GRID_SIZE*c) + r]
    return ret



def hidden_weights_indy(w,digit,ax):
    grid=convert_to_square(w)
    ax=sns.heatmap(grid, cmap='Greys', ax=ax)
    ax.set_title(f"Weight Map for Output Neuron {digit}")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def hidden_weights_super(W): 
    fig, axes = plt.subplots(2,5, figsize=(18,12))
    axes=axes.flatten()
    for i,ax in enumerate(axes):
        ax=hidden_weights_indy(W[i],i, ax)
    plt.suptitle("Output Layer Weight Maps")
    plt.savefig(f"run_{runname}_results/weight_maps.png", bbox_inches='tight')





# SINGLE RUN 

# Runs the model
# Returns the errrors of the run in numpy arrays of length EPOCHS/10 + 1 
def run_model(weights_function, training_function, runname, case):
    try:
        os.mkdir(f"run_{runname}_results/case{case}")
    except FileExistsError: 
        print("File already exists")
    # Create data files and load the data
    get_data(runname)
    X_train, y_train = load_dataset(f"run_{runname}_results/TrainingData{runname}.txt", f"run_{runname}_results/TrainingLabels{runname}.txt")
    X_test,  y_test  = load_dataset(f"run_{runname}_results/TestData{runname}.txt", f"run_{runname}_results/TestLabels{runname}.txt")


    # Training 
    print(f"Case {case}")
    W1, W2 = weights_function(INPUT_LENGTH, HIDDEN_LAYER_NEURONS, OUTPUT_LAYER_NEURONS)
    W1, W2, train_errors, test_errors = training_function(W1, W2, X_train, y_train, X_test, y_test)

    # Plot errors
    plot_errors(train_errors, test_errors,case)

    make_confusion(W1, W2, X_test,y_test, "Test Set Confusion Matrix",case)
    hidden_weights_super(W2)
    # Save the weights at the end
    np.savez(f"run_{runname}_results/case{case}/final_weights{runname}.npz", W1=W1, W2=W2)
    np.savez(f"run_{runname}_results/case{case}/error_history{runname}.npz", train_errors=train_errors, test_errors=test_errors)
    return train_errors, test_errors



# FULL RUN

runname=input("Enter Run Name:")
try:
    os.mkdir(f"run_{runname}_results")
finally:
    train_error, test_error=run_model(init_weights_used, train_just_output, runname, 0)
