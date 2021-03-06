# Team T:  
* Luca Rossi SCIPER 331192,
* Lucas Maximilian Schlenger SCIPER 331031
* Jonas Lyng-Jorgensen SCIPER 334127
Id best submission: 141167

# Packages needed:
1. numpy
2. matplotlib
3. pandas
4. torch
5. torchvision
6. tqdm
7. tensorboard
8. tifffile

# Additional files needed:
trainer.py (provided on moodle)

# Main file: 
train.ipynb

# Remark:
We slightly modified the file trainer.py. Therefore, to run the code you have to manually substitute the file trainer.py which is automatically cloned from the git repository with the file which we provided on moodle.

# Code explanation:

We clone the repository, we download the dataset and we import the file needed. Then we use the package transform of torch vision to create a composition of tranformation (RandomHorizontalFlipping and RandomVerticalFlipping) which we will apply to our training set in order to perform data augmentation. We tried also different types of transformations like adding some random rotation and gaussian blur, however, we had the best performance in term of accuracy just using the two above mentioned random flippings. We set the batch size and we set the path for the cvs files. We applied data augmentation just on the training set because we wanted the validation set to be as similar as possible to the test set so as to use it to tune the best hyperparameters for the model.
Then, we use PatchPairsDataset to have the augmented train images and we use split_dataset to take 80% of them as the training set, then we use again PatchPairsDataset without any transformation to get the non-augmented images and we take the 20% of them as the validation set. Finally we use again PatchPairsDataset without any transformation to create the test data and use DataLoader to load the training, validation and test data.
After the data preprocessing and after having displayed a few images to understand how is our dataset, we are ready to define the architecture of our neural network.
The network is made up of 14 convolutional layers with the same kernel size (=3), stride(=1) and padding(=1), but with an increasing number of output channels. Using this type of hyperparaters allows the size of the image to remain the same (when the convolution was applied). However, 3 maxpooling operations and a final average pooling operation are made in between in order to compress the information of the images and change her dimensions. We also added batchnormalization after each convolution to avoid problem of gradient explosion and vanishing gradient and to assure faster convergence of the stochastic gradient descent, we chose relu as the activation function. After the 14 convolutional layers, a single linear layer provides the final output.
This is not the only network architecture we tried, the commented code after the definition of the above mentioned architecture was the previous trial. The old neural network was made up of 3 convolutional layers with maxpooling in between followed by 3 linear fully connected layers with some dropout in between to reduce overfitting. Despite we changed the model, the old architecture was performing quite good (0.930 accuracy), however, the enormous amount of trainable parameters of the new architeture helped her score way better (0.962 accuracy).
We chose to use the binarycrossentropy loss as the loss (since the problem is a classification one) and we used the BCEWithLogitsLoss() of pytorch which automatically add a sigmoid layer at our neural net. We use the stochastic gradient descent with learning rate 2.5e-2 as the optimizer; we also tried to add some weight decay and momentum but the performance were a bit worse in this case. In addition, we used a scheduler to halve the learning rate each ten epoch in order to have more stable convergence of the stochastic gradient descent. We set the number of epochs to 80. As mentioned before, we slightly modified the file trainer.py so that it saves the model at each epoch, this allows us to select, after the training, the model corresponding to the epoch which have the best validation accuracy. We chose not to save just the model with the best validation accuracy because we wanted to have the possibility to generate submissions for several models without running the training multiple times.
In order to select a particular model to generate the submission it is sufficient to change the parameter date with the name of the folder generated in cloned-repo/project/outputs and to select the epoch of the corresponding model. Finally, we make the prediction and we generate the cvs file.

# How to run the code

Using google colab run the first cell of the file train.ipynb, then substitute the file trainer.py in the folder cloned-repo/project with the one we provided you on moodle and then run the following cells. After the traininig process change the parameters data and epoch to select the model which you use to make your predictions as described in code explanation.