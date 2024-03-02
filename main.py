import classes as cs
import arguments as arg


# Define the training and testing dataset
trainingSet_loader = cs.Dataset(arg.address1).loadData()
testingSet_loader = cs.Dataset(arg.address2).loadData()

# Initialize the neurons
model = cs.Model(trainingSet_loader)

# Train the model using gradient descent
model.train()

# Save model and plot loss-curve
cs.plot_loss_curve(model)

# Make predictions on the testing set
model.test(testingSet_loader)

# test = cs.Test(model, testingSet)
