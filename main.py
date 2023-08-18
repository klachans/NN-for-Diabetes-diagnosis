import pandas as pd
from funcs import *
from network import Network

def main():
    ## Importing data
    pima_tr = pd.read_csv('data\pima_tr.csv',delimiter=' ')
    pima_te = pd.read_csv('data\pima_te.csv',delimiter=' ')
    pima_merged = pd.concat([pima_tr,pima_te],ignore_index=True) 

    ## Normalizing to range (0,1)
    dummy = normalize(pima_merged)
    for header,value in zip(pima_merged.columns.values,dummy):
            pima_merged[header]=value

    pima_tr = pima_merged.iloc[:len(pima_tr),:]
    pima_te = pima_merged.iloc[len(pima_tr):,:]
        
    ## Creating a network with 7 input neurons and one hidden layer with 7 neurons
    ## At this point, we can easily modify the number of neurons in the hidden layer.
    Network_training = Network(7, 7)

    ## Randomly initializing weights from the range (-0.1, 0.1).
    Network_training.randomizeWeights(0.1)

    ## Providing the training data for the network.
    Network_training.feedData(pima_tr)

    ## The number of iterations (epochs) where the network is trained on all shuffled training data.
    ## Value can vary, depending on data, should be matched acordingly to prevent
    ## eg. underfitting or overfitting.
    epochs = 200

    print("---")
    for epoch in range(epochs):
        if epoch % (epochs/5) == 0: ## Changing the number 4 to another value allows us to monitor the network.
            print(f"{100*epoch/epochs}% training complete")
            print("Current results from the training dataset")
            Network_training.checkAccuracy() ## Current results (one iteration with unchanged weights)

        Network_training.oneEpoch() ## One iteration with weight updates

    print("The results from the training dataset: ")
    Network_training.checkAccuracy() ## The results after all iterations
    weights = Network_training.getWeights() ## Collecting weights from the training network

    ## Creating a new network, initializing it with weights from the training network
    ## and providing test data
    Network_testing = Network(7, 7)
    Network_testing.feedWeights(weights[0], weights[1])
    Network_testing.feedData(pima_te)


    print("The results from the test dataset:")
    Network_testing.checkAccuracy()

if __name__ == "__main__":
    main()