from pathlib import Path
from app.hf043_replay import CHANNELS

def test_replay_contract_files_named():
    root=Path('frozen/a5_v1/replay/ardross')
    assert (root/'README.txt').exists()
    assert CHANNELS==['obs','norm','mask','depth','horizontal','vertical','compact','preserve','remove','ambiguous','edge']
