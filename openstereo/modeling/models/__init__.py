from .casnet import CasStereoNet
from .cfnet import CFNet
# from .dsmnet import DSMNet1x
# from .dsmnet import DSMNet2x
from .gwcnet import GwcNet
from .lacgwc import LacGwcNet

try:
    from .ganet import GANet
except ImportError:
    print('GANet is not available')
