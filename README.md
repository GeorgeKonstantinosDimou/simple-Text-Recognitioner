# simple-Text-Recognitioner
A simple text-recognition model which aims to tackle the mentioned problem with a simple architecture

The model consists of two parts, a ResNet-9 backbone and a LSTM responsible for the final character recognition.

The overall project constist of:
  1) Dataset.py which contains an override of the Dataset class responsible for the loading and preprocessing of the images and their initial input into tensors.
  2) Model.py which supports two different architectures, one with a model trained from scratch and one with a pre-trained ResNet18 model.
  3) Backbone.py the first of two parts of the architecture, the ResNet-9.
  4) Recognizer.py the second part a LSTM.
  5) Utils.py because of the usage of CTCLoss as a loss function, special care needs to be attributed to the calculation of accuracy.
  6) Main.py with the remaining functions to train/test the model.
  
The model was trained and tested using IIIT5k dataset, more on Requirements.txt
  
