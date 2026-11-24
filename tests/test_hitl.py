import numpy as np
from music_annotation.hitl_generator import pos
from music_annotation.hitl_reviewer_backend import decode_mask_data_url

def test_positions():
 assert pos(640)==[0]
 assert pos(1000)==[0,320,360]
def test_empty_mask():
 assert decode_mask_data_url('',(3,4)).shape==(3,4)
