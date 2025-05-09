import pickle
from data.download_data import get_data
from models.model import get_model
import keras
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    x_train, x_test, y_train, y_test = get_data(cfg.data.source)

    with open(cfg.data.pickle_path, 'wb') as f:
        pickle.dump((x_train, x_test, y_train, y_test), f)

    model = get_model()
    optimizer = keras.optimizers.Adam(learning_rate=cfg.model.learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    model.fit(x_train, y_train, batch_size=cfg.model.batch_size, epochs=cfg.model.epochs)
    model.save(cfg.model.save_path)
    print(f"Model saved to {cfg.model.save_path}")

if __name__ == "__main__":
    main()
