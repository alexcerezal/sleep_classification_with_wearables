from src_sint.explain_shap import main as explain_main
from src_sint.generate_synthetic_data import main as generate_main
from src_sint.train_random_forest import main as train_main


def main() -> None:
    generate_main()
    train_main()
    explain_main()


if __name__ == "__main__":
    main()
