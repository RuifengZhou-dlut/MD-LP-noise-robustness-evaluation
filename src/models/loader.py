"""Load trained .h5 models and derive the intermediate-layer views needed for evaluation."""
import tensorflow as tf

from ..common.config import MODELS, checkpoint_path
from .trainer import AdamW, CustomWeightDecaySGD, RSquared, WarmUpCosine
from .network import CUSTOM_OBJECTS


def custom_objects():
    objs = {
        "CustomWeightDecaySGD": CustomWeightDecaySGD,
        "AdamW": AdamW,
        "WarmUpCosine": WarmUpCosine,
        "RSquared": RSquared,
    }
    objs.update(CUSTOM_OBJECTS)
    return objs


def load_model(task, model_key, path=None, exp="clean"):
    """Read a trained h5; `exp` picks the default directory (clean -> model/<task>/clean, noise -> model/<task>/noise_trained)."""
    path = path or checkpoint_path(task, model_key, exp)
    return tf.keras.models.load_model(str(path), custom_objects=custom_objects())


def layer_names(model, model_key):
    """Return the list of intermediate layer names."""
    return [model.layers[i].name for i in MODELS[model_key]["l_label"]]


def make_pred_model(model, model_key):
    """Truncated model: from pred_layer to the output, used to compute predictions at eval time."""
    return tf.keras.Model(
        inputs=model.get_layer(MODELS[model_key]["pred_layer"]).output,
        outputs=model.output,
    )
