"""
Ponto de entrada para o Previsor de Preços de Casas.
"""
from views.user_view import UserView

if __name__ == "__main__":
    data_path = "data/house_prices.csv"  # Atualize com o caminho do seu arquivo CSV
    user_view = UserView(data_path)

    print("Previsor de Preços de Casas")

    while True:
        print("\nMenu:")
        print("1. Exibir análise exploratória dos dados")
        print("2. Treinar e avaliar o modelo")
        print("3. Fazer uma previsão")
        print("4. Sair")

        choice = input("Escolha uma opção: ")
        if choice == "1":
            user_view.show_analysis()
        elif choice == "2":
            user_view.train_and_evaluate()
        elif choice == "3":
            user_view.make_prediction()
        elif choice == "4":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")
