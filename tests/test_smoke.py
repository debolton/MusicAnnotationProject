import numpy as np
from music_annotation.v78_engine import prepare_mask, immutable
from music_annotation.policy import effective_print_threshold, provisional_decision

def test_policy_binary_interaction():
    assert effective_print_threshold(.98,.99)==.995
    assert provisional_decision([.999,.5,.001]).tolist()==['PRINT','UNCERTAIN','ANNOTATION']

def test_prepare_mask_and_immutability():
    seed=np.zeros((9,9),np.uint8); seed[4,4]=255; mask=prepare_mask(seed,2); assert mask.sum()>255
    original=np.arange(81,dtype=np.uint8).reshape(9,9); candidate=np.zeros_like(original); out=immutable(original,candidate,mask); assert np.array_equal(out[mask==0],original[mask==0])
