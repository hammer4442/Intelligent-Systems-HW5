import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math 
import time


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
# Number of epochs ran
EPOCHS=200
# Input length 
INPUT_LENGTH = 784
# The number of elements in one row of the Self Organizing Feature Map
GRID_LENGTH = 12
# The initial value of sigma in the neighborhood function
SIGMA_NOT=.5*GRID_LENGTH
# Value of tau for the neighborhood function 
TAU=EPOCHS/math.log(SIGMA_NOT)
# Learning Rate
LEARNING_RATE=0.05
# Number of images from the trainng set looked at per epoch
SUBSET_SIZE=800

# ACTIVATION FUNCTION AND ITS DERIVITIVE


# INITIALIZING DATA AND WEIGHTS

# Reads data from data and label files and returns contents in forms of numpy arrays 
# Returns two arrays:
# Data Matrix: stored as a SUBSET_SIZExINPUT_LENGTH matrix representing the input values with the bias added 
# Label Vector: stored as a SUBSET_SIZE element numpy array with each element being the label of 
# it's corresponding index's digit in the Data Matrix 
def load_dataset(data_file, label_file):
    data = np.loadtxt(data_file)
    labels = np.loadtxt(label_file, dtype=int)
    return data.astype(np.float32), labels

# Initializes the weights and stores them as two 2D numpy arrays
# The arguments to the funciton represent the number of elements 
def init_weights():
    rng = np.random.default_rng(0)
    std_dev1=1
    W1 = rng.normal(0, std_dev1, size=(GRID_LENGTH**2, INPUT_LENGTH)).astype(np.float32)
    return W1

# Initializes an array that holds x and y poitions for correspodning elements in the weight vector
# The positions are stored in GRID_LENGTH^2x2 numpy array
def init_position_matrix():
    pos=np.empty((GRID_LENGTH**2,2), dtype=np.float32)
    i=0
    for x in range(GRID_LENGTH):
        for y in range(GRID_LENGTH):
            pos[i]=[x,y]
            i+=1
    return pos


    



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




# Backpopagates the output from a single image and returns new weights and the changes that occured
# Same as change_weights but fully changes W1
def change_weights(W1, x, t, pos):
    r_star=pos[i_star_func(W1, x)]
    # lamb stands in for Lambda 
    lamb=lamb_func(pos, r_star, t)
    # MAKE THIS 144*785
    # print("LAMB")
    # print(lamb)

    upd_W1=LEARNING_RATE*lamb[:, None]*(x-W1)
    # upd_W1=LEARNING_RATE*lamb*np.array(map(lambda w: x-w, W1))

    new_W1 = upd_W1 + W1
    return new_W1


def i_star_func(W1, x):
    diff=W1-x
    # sums=np.array(map(lambda w: np.sum(w), hold))
    sums= np.sum(diff**2, axis=1)
    return np.argmin(sums)

def i_star_func_test(W1, x):
    diff=W1-x
    # sums=np.array(map(lambda w: np.sum(w), hold))
    sums= np.sum(diff**2, axis=1)
    return np.argmin(sums)

def lamb_func(pos, r_star, t):
    # hold= np.array([np.sum(x - r_star) for x in pos])
    # ret=-1 * (hold**2)
    diff = pos- r_star
    dist=np.sum(diff**2, axis=1)
    sigma=sigma_func(t)
    return np.exp(-dist/(2*sigma**2))

def sigma_func(t):
    return SIGMA_NOT * math.exp(-t/TAU)
    



# Trians the model over a course of epochs equal to EPOCHS
# When the final epoch is run the final weights and two lists of the erros for every 10 epochs is returned
def train(W1, X_train, pos):

    # Runs for a set number of epochs
    # Could easily be changed to a while loop looking at the error 
    for epoch in range(1, EPOCHS + 1):
        # Randomly choose SUBSET_SIZE elements from the training set 
        seed = np.random.choice(len(X_train), size=SUBSET_SIZE, replace=False)
        for i in seed:
            x = X_train[i]
            W1= change_weights(W1, x, epoch, pos)
        if epoch%10==0:
            print(f"Epoch {epoch}")
    return W1




def heatmap_super(W1, X, runname):
    start=time.perf_counter()
    fig, axes = plt.subplots(5,2, figsize=(18,12))
    axes=axes.flatten()
    for i,ax in enumerate(axes):
        ax=heatmap_indy(W1, X[i*100:(i+1)*100],i, ax)
    plt.suptitle("Activity Maps")
    plt.savefig(f"run_{runname}_results/activity_maps.png", bbox_inches='tight')
    end=time.perf_counter()
    total= end-start

    

def heatmap_indy(W1, X, digit, ax):
    winners=np.zeros((GRID_LENGTH,GRID_LENGTH), dtype=int)
    for x in X:
        i_star=i_star_func(W1,x)
        winners[i_star//GRID_LENGTH][i_star%GRID_LENGTH]+=1
    ax=sns.heatmap(winners, cmap='Greys', ax=ax)
    ax.set_title(f"Activity Map for {digit}")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def convert_to_square(arr):
    ret=np.zeros((int(math.sqrt(INPUT_LENGTH)),int(math.sqrt(INPUT_LENGTH))))
    for r in range(int(math.sqrt(INPUT_LENGTH))):
        for c in range(int(math.sqrt(INPUT_LENGTH))):
            ret[r][c]=arr[(int(math.sqrt(INPUT_LENGTH))*c) + r]
    return ret

                       

def featuremap_indy(w, ax):
    grid=convert_to_square(w)
    ax=sns.heatmap(grid, cmap='Greys', cbar=False, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def feautremap_super(W, runname):
    fig, axes = plt.subplots(GRID_LENGTH,GRID_LENGTH)
    axes=axes.flatten()
    for i,ax in enumerate(axes):
        ax=featuremap_indy(W[i],ax)
    plt.suptitle("Feature Maps")
    plt.savefig(f"run_{runname}_results/feature_maps.png")






runname=input("Enter Run Name:")
try:
    os.mkdir(f"run_{runname}_results")
except FileExistsError: 
    print("File already exists")
get_data(runname)
W1 = init_weights()
pos=init_position_matrix()
X_train, y_train = load_dataset(f"run_{runname}_results/TrainingData{runname}.txt", f"run_{runname}_results/TrainingLabels{runname}.txt")
X_test, y_test = load_dataset(f"run_{runname}_results/TestData{runname}.txt", f"run_{runname}_results/TestLabels{runname}.txt")
start=time.perf_counter()
final_W=train(W1, X_train, pos)
end=time.perf_counter()
total= end-start
np.savez(f"run_{runname}_results/final_weights{runname}.npz", W1=final_W)
np.savez("q2weights.npz", W1=final_W)
heatmap_super(final_W, X_test,runname)
feautremap_super(final_W, runname)

















