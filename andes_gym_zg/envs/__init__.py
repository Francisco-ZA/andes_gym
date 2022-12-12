from gym.envs.registration import register
from andes_gym.envs.andes_freq import AndesFreqControl
from andes_gym.envs.andes_freq_2 import AndesPrimaryFreqControl
from andes_gym.envs.andes_freq_3 import AndesPrimaryFreqControl_rocof

register(
    id='AndesFreqControl-v0',
    entry_point='andes_gym:AndesFreqControl',
)

register(
    id='AndesPrimaryFreqControl-v0',
    entry_point='andes_gym:AndesPrimaryFreqControl',
)

register(
    id='AndesPrimaryFreqControl_rocof-v0',
    entry_point='andes_gym:AndesPrimaryFreqControl_rocof',
)
