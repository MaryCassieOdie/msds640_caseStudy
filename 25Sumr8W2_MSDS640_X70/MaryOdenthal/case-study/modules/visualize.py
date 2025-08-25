import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



#Setup class
class DataVisualizer:

    data = []

    
    def __init__(self, data):
        self.data = data


    def compare_genders(self, target, title):
        plt.figure(figsize=(10,7))
        # Plotting the KDE Plot
        sns.kdeplot(self.data.loc[(self.data['SEX']==0),
                    target], color='r', fill=True, label='Female')

        sns.kdeplot(self.data.loc[(self.data['SEX']==1), 
                    target], color='b', fill=True, label='Male')

        plt.xlabel(target)
        plt.ylabel('Probability Density')
        plt.legend()
        plt.title(title)
        plt.show()

    
    def check_balance(self):
        # Create bar chart
        counts = self.data['SEX'].value_counts()
        plt.bar(counts.index, counts.values, color='skyblue')

        # Add labels and title
        plt.xlabel('SEX')
        plt.title('Basic Bar Chart')

        # Show the plot
        plt.show()

