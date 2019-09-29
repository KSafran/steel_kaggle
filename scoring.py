from keras.models import load_model
from src.data_generator import DataGenerator

model = load_model('models/model.h5')
test_ids = pd.read_csv('data/test_images.csv', header=None,
    names=['image'])

valid_gen = DataGenerator(
    'test',
    test_ids['image'],
    batch_size=10,
    shuffle=True)

predictions = model.predict(valid_gen)
