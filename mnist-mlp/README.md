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

<img src="acc_0.png" width="420px" height="280px">
<img src="loss_0.png" width="420px" height="280px">

- Train accuracy: 0.9918
- Test accuracy: 0.9799
- Initial observation: model performs pretty good, but tends to overfitting

## Tasks
- [x] increase/decrease number of nodes in each hidden layer
  - increased num of nodes in each layer to `1024`: 
  
    <img src="acc_1_1.png" width="420px" height="280px">
    <img src="loss_1_1.png" width="420px" height="280px">
    
    - Train accuracy: 0.9923
    - Test accuracy: 0.9798
    - Observation: seems like model memorized training data and tends to overfitting
    
  - decreased num of nodes in each layer to `256`: 
    
    <img src="acc_1_2.png" width="420px" height="280px">
    <img src="loss_1_2.png" width="420px" height="280px">
      
     - Train accuracy: 0.9894
     - Test accuracy: 0.9823
     - Observation: model performs better than previous, overfitting reduced 

- [x] increase/decrease number of hidden layers
  - added +1 more hidden layer of size `256` (activation `relu`, dropout `0.2`): 
    
    <img src="acc_2_1.png" width="420px" height="280px">
    <img src="loss_2_1.png" width="420px" height="280px">
    
      - Train accuracy: 0.9890
      - Test accuracy: 0.9766
      - Observation: model definitely overfits even more
      
  - reduced number of hidden layers so now our model has only one of size `512`:
  
     <img src="acc_2_2.png" width="420px" height="280px">
     <img src="loss_2_2.png" width="420px" height="280px">
    
      - Train accuracy: 0.9937
      - Test accuracy: 0.9817
      - Observation: overfitting was reduced, but it worked worse, than step 1.2

- [x] change dropout layers 
  - removed all dropout layers: 
    
     <img src="acc_3_1.png" width="420px" height="280px">
     <img src="loss_3_1.png" width="420px" height="280px">
  
    - Train accuracy: 0.9947
    - Test accuracy: 0.9751
    - Observation: model overfits heavily, we definitely need regularization
   
   - increased dropout to 0.4 in the first layer and to 0.3 in the second layer: 
    
     <img src="acc_3_2.png" width="420px" height="280px">
     <img src="loss_3_2.png" width="420px" height="280px">
     
     - Train accuracy: 0.9850
     - Test accuracy: 0.9829
     - Observation: overfitting reduced, model performs better
      
- [ ] remove `relu` ativation functions from each layer 

- [ ] remove pre-processing step with dividing by 255 

- [ ] use different optimizer 

- [ ] change batch size

