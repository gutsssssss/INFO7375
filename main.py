import class_file as cf
import numpy as np
import arguments as arg

# Define the training and testing dataset
trainingSet = cf.Dataset(arg.img_height, arg.img_width, arg.num_classes)
trainingSet.loadData(arg.address1)
testingSet = cf.Dataset(arg.img_height, arg.img_width, arg.num_classes)
testingSet.loadData(arg.address2)

# Initialize the neurons
model = cf.Neurons(trainingSet, arg.alpha)

# Train the model using gradient descent
training = cf.Training(trainingSet, model, arg.epochs)
training.run()

# Save model and plot loss-curve
np.savez("model.npz", W=model.W, b=model.b)
cf.plot_loss_curve(training)

# Make predictions on the testing set
test = cf.Test(model, testingSet)
