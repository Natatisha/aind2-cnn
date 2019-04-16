# MNIST data set classification model 
## Initial model 
```
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
```
- Train accuracy: 0.9812
- Test accuracy: 0.9796

## Tasks
- [x] increase/decrease number of nodes in each hidden layer
  - increased num of nodes in each layer to `1024`: 
  
    <img src="acc_1_1.png" width="420px" height="280px">
    <img src="loss_1_1.png" width="420px" height="280px">
    
    - Train accuracy: 0.9923
    - Test accuracy: 0.9798
    - Observation: seems like model memorized training data and tends to overfitting

- [ ] increase/decrease number of hidden layers

- [ ] remove dropout layers 

- [ ] remove `relu` ativation functions from each layer 

- [ ] remove pre-processing step with dividing by 255 

- [ ] use different optimizer 

- [ ] change batch size

