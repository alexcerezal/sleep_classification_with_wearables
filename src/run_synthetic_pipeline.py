from src.explain_shap import main as explain_main
from src.generate_synthetic_data import main as generate_main
from src.train_random_forest import main as train_main


def main() -> None:
    generate_main()
    train_main()
    explain_main()


if __name__ == "__main__":
    main()
