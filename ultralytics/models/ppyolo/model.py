"""
ppyolo model interface
"""
from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info
from .val import PPYOLOValidator
from .predict import DetectionPredictor


class PPYOLO(Model):
    """
    PPYOLO object detection model.
    """

    @property
    def task_map(self):
        """
        Map head to validator
        """
        return {"detect": {"validator": PPYOLOValidator,
                           "predictor": DetectionPredictor}}
