"""
Gerencia a construção e treinamento do modelo de previsão de preços.
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

class HousePriceModel:
    """Gerencia o modelo de previsão de preços de casas."""
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.model = LinearRegression()

    def preprocess_data(self):
        """Pré-processa os dados para treinamento."""
        X = self.data.drop("price", axis=1)
        y = self.data["price"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        """Treina o modelo de machine learning."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Avalia o modelo nos dados de teste."""
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        return mse

    def predict(self, features):
        """Realiza previsões com base nos recursos fornecidos."""
        return self.model.predict([features])