import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_FILE = "sms.tsv"


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Download and read the SMS Spam dataset."""
    if not os.path.exists(path):
        path = tf.keras.utils.get_file(path, DATA_URL)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def prepare_datasets(df: pd.DataFrame, batch_size: int = 32):
    """Split the data and create TensorFlow datasets."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=10000, output_sequence_length=100
    )
    vectorize_layer.adapt(X_train.values)

    def vectorize_text(text, label):
        vectors = vectorize_layer(tf.expand_dims(text, -1))
        return tf.squeeze(vectors, axis=0), label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .map(vectorize_text)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .map(vectorize_text)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds, vectorize_layer


def build_model(vocab_size: int = 10000, embedding_dim: int = 16) -> tf.keras.Model:
    """Create the text classification model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    df = load_data()
    train_ds, test_ds, _ = prepare_datasets(df)
    model = build_model()

    model.fit(train_ds, validation_data=test_ds, epochs=10)
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc:.4f}")

    model.save("spam_model.keras")


if __name__ == "__main__":
    main()
