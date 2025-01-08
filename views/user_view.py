"""
Interage com o usuário e exibe resultados e gráficos.
"""
from models.house_price_model import HousePriceModel
from controllers.data_analysis_controller import DataAnalysisController

class UserView:
    """Gerencia a interação com o usuário."""
    def __init__(self, data_path: str):
        self.model = HousePriceModel(data_path)
        self.analysis = DataAnalysisController(data_path)

    def show_analysis(self):
        """Exibe a análise exploratória dos dados."""
        self.analysis.show_data_info()
        self.analysis.plot_correlation_matrix()

    def train_and_evaluate(self):
        """Treina e avalia o modelo de machine learning."""
        self.model.preprocess_data()
        self.model.train_model()
        mse = self.model.evaluate_model()
        print(f"Erro Médio Quadrado (MSE) no conjunto de teste: {mse}")

    def make_prediction(self):
        """Solicita recursos do usuário e realiza uma previsão."""
        # Certifique-se de que o modelo foi treinado
        if not hasattr(self.model, 'X_train'):
            print("Treinando o modelo antes de realizar a previsão...")
            self.model.preprocess_data()
            self.model.train_model()

        print("Digite os valores para as seguintes características:")
        feature_names = ["location", "area", "bedrooms", "house_age"]
        features = {name: float(input(f"{name.capitalize()}: ")) for name in feature_names}

        # Ordena os valores conforme a ordem das colunas do modelo
        feature_values = [features[name] for name in self.model.X_train.columns]
        prediction = self.model.predict(feature_values)
        print(f"Preço previsto: ${prediction[0]:,.2f}")

