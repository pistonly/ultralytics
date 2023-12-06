'''
hisi om model 
'''
from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info
from .val import HisiValidator
# from .predict import HisiPredictor


class HisiOM(Model):
    """
    PPYOLO object detection model.
    """

    @property
    def task_map(self):
        """
        Map head to validator
        """
        return {"detect": {"validator": HisiValidator,
                           "predictor": None}}
