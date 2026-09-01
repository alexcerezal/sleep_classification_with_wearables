import pandas as pd

CSV_PATH = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\dreamt_epoch_features2.csv"

# Leer CSV
df = pd.read_csv(CSV_PATH)

def mapear_labels(df):
    # Renombrar columna
    df = df.rename(columns={"etiqueta_4_fases": "etiqueta_binaria"})

    # Mapeo binario
    mapping = {
        "W": "Wake",
        "Light_Sleep": "Sleep",
        "Deep_Sleep": "Sleep",
        "REM": "Sleep",
    }

    df["etiqueta_binaria"] = df["etiqueta_binaria"].map(mapping)

    OUTPUT_PATH = r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\dreamt_epoch_features3.csv"
    df.to_csv(OUTPUT_PATH, index=False)

    print("CSV actualizado correctamente.")

def move_column_after(df, column_to_move, reference_column):
    """
    Mueve column_to_move justo después de reference_column,
    manteniendo el resto de columnas en el mismo orden relativo.
    """
    if column_to_move not in df.columns:
        raise ValueError(f"No existe la columna a mover: {column_to_move}")

    if reference_column not in df.columns:
        raise ValueError(f"No existe la columna de referencia: {reference_column}")

    cols = list(df.columns)

    # Quitar la columna que se va a mover para evitar duplicados
    cols.remove(column_to_move)

    # Buscar la posición de la columna de referencia en la lista ya sin duplicados
    ref_idx = cols.index(reference_column)

    # Insertar justo después
    cols.insert(ref_idx + 1, column_to_move)

    return df[cols]

if __name__ == "__main__":

    df = move_column_after(df, "BVP_range", "BVP_std")
    df = move_column_after(df, "ACC_INDEX", "IBI_std")

    df.to_csv(CSV_PATH, index=False)

    print("Columnas reordenadas correctamente.")