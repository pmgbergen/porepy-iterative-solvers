version = "0.0.1"

from .block_linear_system import *
from .mat_utils import *
from .petsc_utils import *
from .plot_linear_system import *
from .porepy_integration import *
from .preconditioners import *

__all__ = []
__all__.extend(block_linear_system.__all__)
__all__.extend(mat_utils.__all__)
__all__.extend(porepy_integration.__all__)
__all__.extend(preconditioners.__all__)
__all__.extend(petsc_utils.__all__)
__all__.extend(plot_linear_system.__all__)
