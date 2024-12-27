import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, filename, learning_rate=0.01, num_iterations=1000):
       
        self.filename = filename
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.teta = None  # Initial teta values will be 0 in the calculated_values function.

    def read_train_datas(self):
        # I am reading the training data file
        train_data = np.array(pd.read_excel(self.filename, sheet_name="TRAINData"), dtype=np.int64)
        # I do not include the column SubjectID
        train_data = train_data[:, 1:]
        # Now bias terms should be added
        ones_column = np.ones((train_data.shape[0], 1))
        train_data = np.hstack((ones_column, train_data))
        return train_data

    def read_test_datas(self):
        # I am reading the test data file
        test_data = np.array(pd.read_excel(self.filename, sheet_name="TESTData"), dtype=np.int64)
        # I do not include the last column in test data. Because all the values are NaN values. Also the first column is excluded
        test_data = test_data[:, 1:-1]
        # Now bias terms should be added
        ones_column = np.ones((test_data.shape[0], 1))
        test_data = np.hstack((ones_column, test_data))
        return test_data

    def calculated_values(self):
        # I am calculating the dot product of training examples and weights
        training_examples = self.read_train_datas()
        self.teta = np.zeros(training_examples.shape[1] - 1)  # Initializing theta weights

        for _ in range(self.num_iterations):
            for i in range(training_examples.shape[0]):
                value = 0
                for j in range(training_examples.shape[1] - 1):  # I don't include the class values
                    value += training_examples[i][j] * self.teta[j]

                class_value = training_examples[i][-1]  # class value of dataset
                if class_value != self.control_statement(value):
                    delta = class_value - self.control_statement(value)
                    self.update_theta_values(training_examples[i], delta)
                    
        print("Theta Values")
        for i in range(len(self.teta)):
            print("Theta",i,": ", self.teta[i])
            


    def control_statement(self, value):
        # I am using a simple thresholding function
        # If value is greater than or equal to 0, then class label is 4
        if value >= 0:
            return 4
        # If value is less than 0, then class label is 2
        else:
            return 2

    def update_theta_values(self, training_example, delta):
        # I am updating the weights of theta values based on learning rate and delta value
        for i in range(len(self.teta)):
            self.teta[i] += self.learning_rate * training_example[i] * delta

    def predict(self, test_data):
        # I am predicting the class labels for the test data without using np.dot
        predictions = []
        for example in test_data:
            value = 0
            # I am calculating the dot product of test example and theta
            for i in range(len(self.teta)):
                value += self.teta[i] * example[i]
            predictions.append(self.control_statement(value))
        return predictions
    
    def write_class_values_of_test_data(self, predicted_class_values):
        # I load the original test data from the specified Excel file
        test_data = pd.read_excel(self.filename, sheet_name='TESTData')
        
        # I add the predicted values to a new column or replace the existing 'Class' column
        test_data['Class'] = predicted_class_values
        
        # I write the updated data back to the same Excel file
        with pd.ExcelWriter(self.filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            test_data.to_excel(writer, sheet_name='TESTData', index=False)


def main():

    filename = "DataForPerceptron.xlsx"  

    # I am creating an instance of the class Perceptron
    perceptron = Perceptron(filename)

    # Here I am producting the teta values and training datas.
    perceptron.calculated_values()

    # I have done my model. Now I am testing my model with test datas
    test_data = perceptron.read_test_datas()
    predictions = perceptron.predict(test_data)

    # I write the predicted class values to the Excel file
    perceptron.write_class_values_of_test_data(predictions)
    
    for i in range(len(predictions)):
        print("SubjectID ",550+i+1,": ", predictions[i])
    

if __name__ == "__main__":
    main()
