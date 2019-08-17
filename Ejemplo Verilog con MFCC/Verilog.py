from python_speech_features import mfcc 
from python_speech_features import logfbank
from python_speech_features import base
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import itertools
from scipy import linalg
import matplotlib as mpl

def mfcc_wav(file):
    (rate,sig) = wav.read(file)
    mfcc_feat = mfcc(sig,rate,nfft=512,appendEnergy=True)
    return mfcc_feat

mfcc_feat=mfcc_wav("C:\\Users\\BurgerBucks\\Documents\\Proyectos\\Correlacion-FFT-y-MFCC\\Ejemplo Verilog con MFCC\\mundo.wav")
plt.hist(mfcc_feat.T[5], bins=40)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

WINDOW_SIZE=20
mundo_size=mfcc_feat.shape[0]-WINDOW_SIZE+1
mundo=np.array([mfcc_feat[i:i+WINDOW_SIZE].reshape(1,WINDOW_SIZE*mfcc_feat.shape[1])[0] for i in range(mundo_size)])

mundo.shape

mundo_labels=np.array([[1]]*mundo.shape[0])

mundo_labels[0]

mundo_labels.shape

mfcc_feat=mfcc_wav("C:\\Users\\BurgerBucks\\Documents\\Proyectos\\Correlacion-FFT-y-MFCC\\Ejemplo Verilog con MFCC\\homero.wav")

homero_size=mfcc_feat.shape[0]-WINDOW_SIZE+1
homero=np.array([mfcc_feat[i:i+WINDOW_SIZE].reshape(1,WINDOW_SIZE*mfcc_feat.shape[1])[0] for i in range(homero_size)])

homero.shape

homero[0].shape

homero_labels=np.array([[0]]*homero.shape[0])

homero_labels.shape

mundo_labels.shape

train_labels=np.vstack([homero_labels,mundo_labels])

train_set=np.vstack([homero,mundo])

train_set.shape

train_labels.shape

mfcc_feat=mfcc_wav("C:\\Users\\BurgerBucks\\Documents\\Proyectos\\Correlacion-FFT-y-MFCC\\Ejemplo Verilog con MFCC\\mundo_test_1.wav")
mundo_size=mfcc_feat.shape[0]-WINDOW_SIZE+1
mundo=np.array([mfcc_feat[i:i+WINDOW_SIZE].reshape(1,WINDOW_SIZE*mfcc_feat.shape[1])[0] for i in range(mundo_size)])
mundo_labels=np.array([[1]]*mundo.shape[0])
mfcc_feat=mfcc_wav("C:\\Users\\BurgerBucks\\Documents\\Proyectos\\Correlacion-FFT-y-MFCC\\Ejemplo Verilog con MFCC\\homero_test.wav")
homero_size=mfcc_feat.shape[0]-WINDOW_SIZE+1
homero=np.array([mfcc_feat[i:i+WINDOW_SIZE].reshape(1,WINDOW_SIZE*mfcc_feat.shape[1])[0] for i in range(homero_size)])
homero_labels=np.array([[0]]*homero.shape[0])
test_labels=np.vstack([homero_labels,mundo_labels])
test_set=np.vstack([homero,mundo])

mfcc_feat=mfcc_wav("C:\\Users\\BurgerBucks\\Documents\\Proyectos\\Correlacion-FFT-y-MFCC\\Ejemplo Verilog con MFCC\\mundo_test_2.wav")
mundo_size=mfcc_feat.shape[0]-WINDOW_SIZE+1
mundo=np.array([mfcc_feat[i:i+WINDOW_SIZE].reshape(1,WINDOW_SIZE*mfcc_feat.shape[1])[0] for i in range(mundo_size)])
mundo_labels=np.array([[1]]*mundo.shape[0])
mfcc_feat=mfcc_wav("C:\\Users\\BurgerBucks\\Documents\\Proyectos\\Correlacion-FFT-y-MFCC\\Ejemplo Verilog con MFCC\\homero_test_2.wav")
homero_size=mfcc_feat.shape[0]-WINDOW_SIZE+1
homero=np.array([mfcc_feat[i:i+WINDOW_SIZE].reshape(1,WINDOW_SIZE*mfcc_feat.shape[1])[0] for i in range(homero_size)])
homero_labels=np.array([[0]]*homero.shape[0])
test2_labels=np.vstack([homero_labels,mundo_labels])
test2_set=np.vstack([homero,mundo])

import keras
from keras.models import Sequential
from keras.layers import Dense
batch_size = 128
epochs = 12
model=Sequential()
model.add(Dense(30,input_shape=(260,)))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()

model.fit(x=train_set,y=train_labels,validation_data=[test_set,test_labels],epochs=1)

