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
      
- [x] remove `relu` ativation functions from each layer 
     
     <img src="acc_4_1.png" width="420px" height="280px">
     <img src="loss_4_1.png" width="420px" height="280px">
     
     - Train accuracy: 0.9114
     - Test accuracy: 0.9198
     - Observation: everything looks ok in train plots but test graph looks really unstable. We probably should add at least some activation function. 

- [x] remove pre-processing step with dividing by 255 
          
     <img src="acc_5_1.png" width="420px" height="280px">
     <img src="loss_5_1.png" width="420px" height="280px">
     
     - Train accuracy: 0.1904
     - Test accuracy: 0.3039
     - Observation: looks like something went really wrong. Loss was decreasing at first, but then it started increasing dramatically. So probably our data is skewed in a way that it's hard or even impossible for gradient descent to find global mininum. 

- [X] use different optimizer 
  - Changed `adam` to `sgd`: 
     
     <img src="acc_6_1.png" width="420px" height="280px">
     <img src="loss_6_1.png" width="420px" height="280px">
     
     - Train accuracy: 0.9125
     - Test accuracy: 0.9350
     - Observation: Overfitting reduced but overall performance looks worse (loss function decrease is steady but too slow). Probably 10 epochs is not enough for this optimizer. 
     
   - Changed `adam` to `rms_prop`:
     
     <img src="acc_6_2.png" width="420px" height="280px">
     <img src="loss_6_2.png" width="420px" height="280px">
     
     - Train accuracy: 0.9923
     - Test accuracy: 0.9819
     - Observation: good performance, but model still overfits

- [x] change batch size:
  - increased from `128` to `256`:
       
     <img src="acc_7_1.png" width="420px" height="280px">
     <img src="loss_7_1.png" width="420px" height="280px">
  
    - Training time 8-15 sec per epoch
    - Train accuracy: 0.9915
    - Test accuracy: 0.9817
    
  - decreased from `128` to `64` :
  
     <img src="acc_7_2.png" width="420px" height="280px">
     <img src="loss_7_2.png" width="420px" height="280px">
  
    - Training time 18-27 sec per epoch
    - Train accuracy: 0.9909
    - Test accuracy: 0.9801
    
   - Observation: batch GD learns faster than stochastic (+ the one with bigger batch is faster than the other with smaller batch size), 256 batch size is OK for my computer, but it also could and should be tuned (because if it's too big -> OME).
