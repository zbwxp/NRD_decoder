from .fcn_head import FCNHead
from .bilinear_pad_head_fast import BilinearPADHead_fast
from .segformer_head import SegFormerHead

__all__ = ['BilinearPADHead_fast',
           'FCNHead', 'SegFormerHead']
