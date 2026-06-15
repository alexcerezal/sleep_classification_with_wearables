import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from src_sint.explain_shap import main as explain_main
#from src_sint.generate_synthetic_data import main as generate_main
#from src_sint.train_random_forest import main as train_main


def main() -> None:
    #generate_main()
    #train_main()
    #explain_main()
    print("Synthetic pipeline executed successfully.")


if __name__ == "__main__":
    #main()
    df = pd.read_csv(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\synthetic_sleep_features.csv")
    df = df.drop(columns=["subject_id", "epoch", "label"])

    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
    plt.title('Correlation Matrix')
    plt.show()