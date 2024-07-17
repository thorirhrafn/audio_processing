import librosa
import librosa.display
from ipywidgets import IntProgress
import pandas as pd
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay # to measure how good the model is
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_timestamps(signal, sr):
    """
    Creates an array of timestamps in seconds based off of the sample rate.
    This will be used for the x-axis of the waveform plot.
    """
    timestamps = np.arange(len(signal))
    timestamps = timestamps / sr
    return timestamps

def create_mfcc():    
    data = pd.read_csv('Data/features_30_sec.csv')
    #Preprocessing - geta mfcc values from each audio file and save as an image

    # count = 0
    printProgressBar(0, len(data), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for index, row in data.iterrows():
        if row['filename'] != 'jazz.00054.wav':   # one file is corrupt
            filepath = "Data/genres_original/" + row['label'] + "/" + row['filename']
            audio_signal, sample_rate =  librosa.load(filepath)
            mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)

            name = row['filename']
            name = name.split('.')
            
            out_path = 'mfcc/' + row['label'] + '/' + name[1] + '.png' 
            img = librosa.display.specshow(mfcc)
            plt.savefig(out_path)
                # count += 1
                # if count > 5:
                #    break

            printProgressBar(index + 1, len(data), prefix = 'Progress:', suffix = 'Complete', length = 50)


def create_mfcc_delta():
    data = pd.read_csv('Data/features_30_sec.csv')

    count = 0
    printProgressBar(0, len(data), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for index, row in data.iterrows():
        if row['filename'] != 'jazz.00054.wav':   # one file is corrupt
            filepath = "Data/genres_original/" + row['label'] + "/" + row['filename']
            audio_signal, sample_rate =  librosa.load(filepath)
            mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
            delta_mfcc  = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            plt.subplot(3,1,1)
            librosa.display.specshow(mfcc)
            plt.subplot(3,1,2)
            librosa.display.specshow(delta_mfcc)
            plt.subplot(3,1,3)
            librosa.display.specshow(delta2_mfcc, sr=sample_rate)

            name = row['filename']
            name = name.split('.')
        
            out_path = 'mfcc_delta/' + row['label'] + '/' + name[1] + '.png' 
            plt.savefig(out_path)
            # count += 1
            # if count > 5:
            #    break

            printProgressBar(index + 1, len(data), prefix = 'Progress:', suffix = 'Complete', length = 50)    


def main():
    create_mfcc_delta()
        

if __name__ == '__main__':
    main()