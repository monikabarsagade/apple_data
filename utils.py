import pickle
import numpy as np

# this function is use to predict apple quality, added by Nitin.
def get_apple_quality(Size, Weight, Sweetness,Crunchiness,Juiciness,Ripeness,Acidity):
    model_file_path = r"artifact\knn.pkl"

    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)

    test_array = np.array([[Size, Weight, Sweetness,Crunchiness,Juiciness,Ripeness,Acidity]])
    apple_quality = model.predict(test_array)[0]

    return apple_quality
