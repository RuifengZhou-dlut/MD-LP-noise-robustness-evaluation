from train import WarmUpCosine, CustomWeightDecaySGD, AdamW
from network import HierarchicalConvEmbedding, TransformerBlock, AddPositionEmbedding, MLP, MixerBlock
import tensorflow as tf

def load_VGG():
    model = tf.keras.models.load_model('VGG_11.h5',custom_objects={
        'CustomWeightDecaySGD': CustomWeightDecaySGD,
        'WarmUpCosine': WarmUpCosine
    })
    return model

def load_Res():
    model = tf.keras.models.load_model('Res_18.h5',custom_objects={
        'CustomWeightDecaySGD': CustomWeightDecaySGD,
        'WarmUpCosine': WarmUpCosine
    })
    return model

def load_ViT():
    model = tf.keras.models.load_model('ViT_8.h5',custom_objects={
        'AdamW': AdamW,
        'WarmUpCosine': WarmUpCosine,
        'HierarchicalConvEmbedding':HierarchicalConvEmbedding,
        'TransformerBlock':TransformerBlock,
        "AddPositionEmbedding": AddPositionEmbedding
    })
    return model

def load_Mix():
    model = tf.keras.models.load_model('Mix_8.h5',custom_objects={
        'AdamW': AdamW,
        'WarmUpCosine': WarmUpCosine,
        'HierarchicalConvEmbedding':HierarchicalConvEmbedding,
        'MLP':MLP,
        "MixerBlock": MixerBlock,
    })
    return model