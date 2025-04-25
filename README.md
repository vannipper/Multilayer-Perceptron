# Multilayer-Perceptron
A class-based Python implementation of a basic multilayer perceptron written by Van Nipper.

Developed for the 2024-2025 senior design project "Computer Vision for Classroom Attendance"

HOW TO TRAIN A MODEL:
1. Ensure the scipy and numpy libraries are installed by doing "pip install numpy scipy"
2. In your Python program, type "from nn import MLP"
3. Initialize the MLP using the number of input nodes as the sole parameter (ex.) "nn = MLP(26)"
4. Using the example MLP object above append a layer by using the addLayer function with the number of , do nn.addLayer()
5. Train the MLP using the train function. Do "nn.train(X_data, Y_data, num_epochs, learning_rate, X_test, Y_test)"
6. Save the MLP to a file using the save function (ex.) "nn.save(filename)"

HOW TO USE PRE-TRAINED MODEL:
1. Initialize a new MLP object with the correct number of input layers (ex.) "nn = MLP(26)"
2. Use the load function to load the file of weights and biases (ex.) "nn.load(filename)"
3. Input a set of data using the feedForward funciton (ex.) "nn.feedForward(alphabet)". This function returns the vector of values in the output layer of the neural network.
