from .base import BaseDetector

class RboxBaseDetector(BaseDetector):
    """Base class for detectors"""

    def __init__(self):
        super(RboxBaseDetector, self).__init__()

    @property
    def with_rbox(self):
        return hasattr(self, 'rbox_head') and self.rbox_head is not None
