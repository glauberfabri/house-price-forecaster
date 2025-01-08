"""
Realiza a análise exploratória dos dados.
"""
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalysisController:
    """Realiza análises exploratórias nos dados."""
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)

    def show_data_info(self):
        """Exibe informações gerais sobre os dados."""
        print(self.data.info())
        print(self.data.describe())

    def plot_correlation_matrix(self):
        """Plota a matriz de correlação dos dados."""
        correlation = self.data.corr()
        plt.figure(figsize=(10, 8))
        plt.title("Matriz de Correlação")
        plt.imshow(correlation, cmap="coolwarm", interpolation="none")
        plt.colorbar()
        plt.show()
