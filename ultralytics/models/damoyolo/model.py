"""
ppyolo model interface
"""
from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info
from .val import DAMOYOLOValidator


class DAMOYOLO(Model):
    """
    PPYOLO object detection model.
    """

    @property
    def task_map(self):
        """
        Map head to validator
        """
        return {"detect": {"validator": DAMOYOLOValidator}}
