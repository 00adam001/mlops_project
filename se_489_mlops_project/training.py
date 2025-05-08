import pickle
from data import get_data
from models import get_model
import keras

x_train, x_test, y_train, y_test = get_data('sudoku.csv')

with open('sudoku_data.pickle', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test), f)
    
model = get_model()

adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

model.fit(x_train, y_train, batch_size=32, epochs=2)

model.save('sudoku_model.h5')  