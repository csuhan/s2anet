from .dota_obb import DotaOBBDataset
from .registry import DATASETS


@DATASETS.register_module
class HRSC2016Dataset(DotaOBBDataset):
    CLASSES = ('ship',)