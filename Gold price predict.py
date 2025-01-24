import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

class GoldPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gold Price Prediction")
        self.root.geometry("600x400")

        # Button for loading data
        self.load_button = tk.Button(self.root, text="Load CSV Data", command=self.load_data)
        self.load_button.pack(pady=20)

        # Button for training the model
        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.pack(pady=20)

        # Button for making predictions
        self.predict_button = tk.Button(self.root, text="Make Predictions", command=self.make_predictions, state=tk.DISABLED)
        self.predict_button.pack(pady=20)

        # Label to display results
        self.result_label = tk.Label(self.root, text="Results will be displayed here.")
        self.result_label.pack(pady=20)

        # Initialize variables for data and model
        self.gold_data = None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        self.regressor = None

    def load_data(self):
        # Open file dialog to load CSV
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
        if file_path:
            self.gold_data = pd.read_csv(file_path)
            self.gold_data['Date'] = pd.to_datetime(self.gold_data['Date'], format='%m/%d/%Y')
            self.result_label.config(text="Data Loaded Successfully!")
            self.train_button.config(state=tk.NORMAL)

    def train_model(self):
        if self.gold_data is not None:
            # Prepare the data
            X = self.gold_data.drop(['Date', 'GLD'], axis=1)
            Y = self.gold_data['GLD']
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
            self.regressor = RandomForestRegressor(n_estimators=100)
            self.regressor.fit(self.X_train, self.Y_train)
            self.result_label.config(text="Model Trained Successfully!")
            self.predict_button.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", "Please load data first.")

    def make_predictions(self):
        if self.regressor:
            test_data_prediction = self.regressor.predict(self.X_test)
            error_score = metrics.r2_score(self.Y_test, test_data_prediction)

            # Display error score
            self.result_label.config(text=f"R squared error: {error_score:.4f}")
            
            # Plotting the results
            plt.plot(self.Y_test.values, color='blue', label='Actual Value')
            plt.plot(test_data_prediction, color='green', label='Predicted Value')
            plt.title('Actual Price vs Predicted Price')
            plt.xlabel('Number of values')
            plt.ylabel('GLD Price')
            plt.legend()
            plt.show()

# Set up the main Tkinter window
root = tk.Tk()
app = GoldPredictionApp(root)
root.mainloop()
