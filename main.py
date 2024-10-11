# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
import os


class IrisEDA:
    def __init__(self):
        # Load the Iris dataset
        self.iris_sklearn = load_iris()
        self.iris_df = pd.DataFrame(data=self.iris_sklearn.data, columns=self.iris_sklearn.feature_names)
        self.iris_df['species'] = self.iris_sklearn.target
        self.iris_df['species'] = self.iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        # Create a directory to save plots
        self.output_dir = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def prepare_data(self):
        """
        Prepares the data by splitting it into training and testing sets and scaling the features.
        Returns: X_train_scaled, X_test_scaled, y_train, y_test
        """
        # Feature extraction (dropping the 'species' column)
        X = self.iris_df.drop('species', axis=1)
        y = self.iris_df['species']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scaling Data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def summary_statistics(self):
        # 1. Summary Statistics
        print("Summary Statistics:")
        print(self.iris_df.describe())

    def missing_values(self):
        # 2. Checking for missing values
        print("\nMissing Values:")
        print(self.iris_df.isnull().sum())

    def correlation_matrix(self):
        # 3. Correlation Matrix (excluding the 'species' column)
        print("\nCorrelation Matrix:")
        correlation_matrix = self.iris_df.drop('species', axis=1).corr()  # Exclude the categorical column
        print(correlation_matrix)

    def visualize_boxplot(self):
        # 4. Boxplot to visualize outliers
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.iris_df)
        plt.title('Boxplot of Iris Features')

        # Save the boxplot as a PNG file
        plt.savefig(os.path.join(self.output_dir, "boxplot_iris_features.png"))
        plt.show()

    def pca_visualization(self):
        """
        Perform PCA on the training data and visualize the result.
        """
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data()

        # PCA
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)

        # Visualizing the PCA result
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=pd.Categorical(y_train).codes, cmap='viridis', s=60)
        plt.title('PCA of Iris Dataset (Training Set)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

        # Save the PCA plot as a PNG file
        plt.savefig(os.path.join(self.output_dir, "pca_iris_dataset.png"))
        plt.show()

    def train_random_forest(self):
        """
        Train a Random Forest classifier and evaluate it.
        """
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data()

        # Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)


# Create an instance of the IrisEDA class
iris_analysis = IrisEDA()

# Call methods to perform EDA and save images
iris_analysis.summary_statistics()
iris_analysis.missing_values()
iris_analysis.correlation_matrix()
iris_analysis.visualize_boxplot()
iris_analysis.pca_visualization()
iris_analysis.train_random_forest()