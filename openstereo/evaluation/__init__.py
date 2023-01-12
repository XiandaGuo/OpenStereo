from numpy import set_printoptions

from .metric import mean_iou

set_printoptions(suppress=True, formatter={'float': '{:0.2f}'.format})
