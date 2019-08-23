import pytest
import numpy as np
import pandas as pd
from src.image_utils import show_mask, id_to_mask, extract_imageid, get_ids 

def test_show_mask():
    mask = show_mask('0 10 100 5') 
    assert mask.shape == (256, 1600)
    assert mask.sum() == 15
    assert all(mask[:10, 0] == np.array([1.] * 10))

def test_id_to_mask():
    mask = id_to_mask('0002cc93b.jpg')
    assert mask.shape == (256, 1600, 4)
    assert mask.sum() == 4396
   
def test_extract_imageid():
    image_id = extract_imageid('235332.jpg_1')
    assert image_id == '235332.jpg'

def test_get_ids():
    ids = get_ids(pd.DataFrame({'ImageId_ClassId': ['a_b', 'c_d']}))
    assert ids == ['a', 'c']
