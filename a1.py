
import librosa
from ipywidgets import IntProgress
import pandas as pd
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay # to measure how good the model is
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


from model import SLP_Model

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def extract_features(dataset, dir, mean=False):
    """
    Extracts MFCCs from the audio files in the dataset.
    Takes the mean of the MFCCs over time if mean=True (1D array), otherwise returns MFCCs for each frame (2D array).
    Returns a list of MFCCs and a list of labels.
    """
    X = []
    X_mean = []
    y = []
    emotions = ['amused', 'angry', 'disgusted', 'drunk', 'neutral', 'sleepy', 'surprised'] #, 'whisper']
    label = -1

    for emotion in emotions:
        label += 1
        for index, row in dataset.iterrows():
            filepath = emotion + "/" + row["id"] + ".wav"
            signal, sr = librosa.load(dir + "/" + filepath)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            #mfcc_deltas = librosa.feature.delta(mfccs) # delta features
            #mfcc_deltas2 = librosa.feature.delta(mfccs, order=2) # delta-delta features
            #X.append(np.concatenate((mfccs.mean(axis=1), mfcc_deltas.mean(axis=1), mfcc_deltas2.mean(axis=1)))) # averaging the features over the file and concatenating them
            if mfccs.shape[1] > 250:
                # only using the first 250 windows
                mfccs_subset = mfccs[:, 0:250]
            else:
                # if too short, add zero padding to mfccs_subset until you get 250 windows
                mfccs_subset = np.pad(mfccs, ((0, 0), (0, 250 - mfccs.shape[1])), 'constant', constant_values=0)
            # flatten the 2d array mfccs_subset to 1d
            mfccs_subset = mfccs_subset.flatten()
            # also trying just averaging the mfccs over time
            mfccs_mean = mfccs.mean(axis=1)
            X_mean.append(mfccs_mean)
            X.append(mfccs_subset)
            y.append(label)
        
    return X, X_mean, y


def process_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    train_input = torch.tensor(X_train, dtype=torch.float)
    train_target = torch.tensor(y_train, dtype=torch.float)
    val_input = torch.tensor(X_val, dtype=torch.float)
    val_target = torch.tensor(y_val, dtype=torch.float)
    test_input = torch.tensor(X_test, dtype=torch.float)
    test_target = torch.tensor(y_test, dtype=torch.float)

    trains_ds = TensorDataset(train_input, train_target)
    validation_ds = TensorDataset(val_input, val_target)
    test_ds = TensorDataset(test_input, test_target)

    train_dl = DataLoader(trains_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(validation_ds, batch_size=32, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_dl, val_dl, test_dl


def train(model, train_dl, val_dl):
    # train the model

    t = 0
    epochs = []
    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #, weight_decay=1e-5)
    criterion = nn.NLLLoss()

    for epoch in range(25):
        train_correct = 0
        train_total = 0
        train_loss = 0
        val_correct = 0
        val_total = 0
        val_loss = 0
        epochs.append(epoch)

        model.train()
        for train_data in train_dl:
            X_train, y_train = train_data
            y_train = y_train.type(torch.LongTensor)
            optimizer.zero_grad() # clear gradient information.
            output_train = model(X_train)  # forward through network

            for index, value in enumerate(output_train):
                if torch.argmax(value) == y_train[index]:
                    train_correct += 1
                train_total += 1

            loss = nn.functional.nll_loss(output_train, y_train)  # loss/cost/error function
            loss.backward() # back propagation (computation)
            optimizer.step() # update weights.
            train_loss += loss.item()
        # save loss and accuracy for this epoch
        # print(train_loss/len(train_dl))
        total_train_loss.append(train_loss/len(train_dl))
        # print(round(train_correct/train_total, 3))
        total_train_acc.append(round(train_correct/train_total, 3))

        print(f'training loss: {loss.data}')
        print(f'training acc: {round(train_correct/train_total, 3)}')
        print("----------")

    
        model.eval()
        with torch.no_grad(): 
            for val_data in val_dl:
                X_val, y_val = val_data
                y_val = y_val.type(torch.LongTensor)
                output_val = model(X_val)

                for index, value in enumerate(output_val):
                    if torch.argmax(value) == y_val[index]:
                        val_correct += 1
                    val_total += 1

                loss = nn.functional.nll_loss(output_val, y_val)  # loss/cost/error function
                val_loss += loss.item()
            total_val_loss.append(val_loss/len(val_dl))  
            total_val_acc.append(round(val_correct/val_total, 3))


    plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)
    plt.plot(epochs, total_train_acc, label="Training Accuracy")
    plt.plot(epochs, total_val_acc, label="Validation Accuracy")
    plt.title("Training and validation")
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.legend(["Training", "Validation"])

    plt.subplot(2,1,2)
    plt.plot(epochs, total_train_loss, label="Training Loss")
    plt.plot(epochs, total_val_loss, label="Validation Loss")
    plt.ylabel('Loss')
    plt.ylim(0,3)
    plt.xlabel('Epochs')
    plt.legend(["Training", "Validation"])

    plt.show()

    return model


def test_model(model, test_dl):
    # Evaluate the trained network.

    total = 0
    correct = 0
    y_pred = []
    y_target = []

    with torch.no_grad():   # No need for keepnig track of necessary changes to the gradient.
        for data in test_dl:
            X, y = data
            output = model(X)

            for index, value in enumerate(output):
                pred = torch.argmax(value) 
                target = y[index]
                # print(f'Predicion: {pred} , Target: {target}')
                y_pred.append(pred)
                y_target.append(target) 
                if pred == target:
                    correct += 1
                total += 1 

            accuracy = 100*correct/total   

    # print(f'Correct: {correct}')
    # print(f'Total: {total}')
    print('Accuracy:', accuracy)     
    print("----------")

    # y_target = y_target.squeeze(0)  
    # print(y_target)
    # print(y_pred)

    classes = ['amused', 'angry', 'disgusted', 'drunk', 'neutral', 'sleepy', 'surprised']
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_target, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    print(classification_report(y_target, y_pred))
        


def main():
    data = pd.read_csv('emotional/emotional/thorsten-emotional-metadata.csv', sep="|", dtype="string", names=['id','sentence'])
    print(len(data))
    print(data.head())
    for col in data.columns:
        print(col)

    dir = 'emotional/emotional'    

    X, X_mean, y = extract_features(data, dir=dir, mean=True)
    X = np.asarray(X)
    y = np.asarray(y)
    print(X.shape[1])
    print(y.shape)

    train_dl, val_dl, test_dl = process_data(X, y)

    model = SLP_Model()
    model = train(model, train_dl, val_dl)
    test_model(model, test_dl)

if __name__ == '__main__':
    main()