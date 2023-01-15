import numpy as np
import tensorflow as tf

housePrices_model = tf.keras.models.load_model('model.h5')

houseData_for_predict = {'Type': np.array(['SFH', 'SFH', 'SFH', 'Condo', 'Condo']),
    'houseEra': np.array(['recent', '19A', '20A', '20A', '19B']),
    'Area': np.array([8410, 1400, 1500, 1500, 1600]),
    'Zip': np.array(['60062', '60062', '60076', '60076', '60202']),
    'Rooms': np.array([16, 6, 7, 7, 7]),
    'FullBaths': np.array([6.0, 2.0, 2.0, 2.5, 2.0]),
    'HalfBaths': np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
    'BsmtBth': np.array(['Yes', 'No', 'No', 'No', 'No']),
    'Beds': np.array([5, 3, 3, 3, 3]),
    'BsmtBeds': np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    'GarageSpaces': np.array([3, 2, 0, 0, 0])}

houseData_for_predict = {'Type': np.array(['SFH']),
    'houseEra': np.array(['recent']),
    'Area': np.array([8410]),
    'Zip': np.array(['60062']),
    'Rooms': np.array([16]),
    'FullBaths': np.array([6.0]),
    'HalfBaths': np.array([0.0]),
    'BsmtBth': np.array(['Yes']),
    'Beds': np.array([5]),
    'BsmtBeds': np.array([1.0]),
    'GarageSpaces': np.array([3])}

result = housePrices_model.predict(houseData_for_predict)
print(result)