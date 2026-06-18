import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def build_vggnet16(NN, num_class, input_shape=(32, 32, 3)):
    model = keras.models.Sequential()
    model.add(layers.Conv2D(filters=NN[0], kernel_size=(3, 3), padding='same'
                            , input_shape=input_shape, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 2
    model.add(layers.Conv2D(NN[1], (3, 3), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 5
    model.add(layers.Conv2D(NN[2], (3, 3), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Conv2D(NN[3], (3, 3), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 10
    model.add(layers.Conv2D(NN[4], (3, 3), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Conv2D(NN[5], (3, 3), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 15
    model.add(layers.Conv2D(NN[6], (3, 3), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Conv2D(NN[7], (3, 3), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # 25
    model.add(layers.Flatten())
    model.add(layers.Dense(NN[7]))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Dense(NN[7]))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.ReLU())
    model.add(layers.Dense(num_class))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def conv_bn_relu(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def residual_block(x, filters, downsample=False):
    shortcut = x
    strides = 2 if downsample else 1
    x = conv_bn_relu(x, filters, 3, strides)
    x = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if downsample or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same',use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    return tf.keras.layers.ReLU()(x)

def build_resnet18(NN, num_classes, input_shape=(32,32,3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = conv_bn_relu(inputs, NN[0], 3)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = residual_block(x, NN[1])
    x = residual_block(x, NN[2])
    x = residual_block(x, NN[3], downsample=True)
    x = residual_block(x, NN[4])
    x = residual_block(x, NN[5], downsample=True)
    x = residual_block(x, NN[6])
    x = residual_block(x, NN[7], downsample=True)
    x = residual_block(x, NN[8])
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes,activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

class HierarchicalConvEmbedding(layers.Layer):
    """
    Conv embedding: use several stride-2 conv layers to downsample the image,
    then project to embed_dim and flatten to tokens.

    Example:
        image_size=32, num_downsamples=2
        32 -> 16 -> 8
        so num_patches = 8 * 8 = 64

    Input:
        (B, H, W, C)
    Output:
        (B, N, D)
    """
    def __init__(self,
                 image_size=32,
                 embed_dim=192,
                 channels=(64, 128),
                 num_downsamples=2,
                 use_bn=True,
                 act="gelu",
                 **kwargs):
        super().__init__(**kwargs)

        assert image_size % (2 ** num_downsamples) == 0, \
            "image_size must be divisible by 2**num_downsamples"

        self.image_size = int(image_size)
        self.embed_dim = int(embed_dim)
        self.channels = tuple(int(c) for c in channels)
        self.num_downsamples = int(num_downsamples)
        self.use_bn = bool(use_bn)
        self.act = str(act)

        self.out_size = self.image_size // (2 ** self.num_downsamples)
        self.num_patches = self.out_size * self.out_size

        self.layers_list = []
        for i in range(self.num_downsamples):
            c = self.channels[min(i, len(self.channels) - 1)]
            self.layers_list.append(
                layers.Conv2D(
                    filters=c,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    use_bias=not self.use_bn,
                    name=f"stem_conv_{i}"
                )
            )
            if self.use_bn:
                self.layers_list.append(
                    layers.BatchNormalization(name=f"stem_bn_{i}")
                )
            self.layers_list.append(
                layers.Activation(self.act, name=f"stem_act_{i}")
            )

        self.proj = layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name="stem_proj"
        )

        self.reshape = layers.Reshape((self.num_patches, self.embed_dim), name="stem_reshape")

    def call(self, x, training=None):
        for layer in self.layers_list:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)

        x = self.proj(x)       # (B, H', W', D)
        x = self.reshape(x)    # (B, N, D)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "embed_dim": self.embed_dim,
            "channels": self.channels,
            "num_downsamples": self.num_downsamples,
            "use_bn": self.use_bn,
            "act": self.act,
        })
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        key_dim = embed_dim // num_heads
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(int(embed_dim * mlp_ratio), activation='gelu'),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, x, training=False):
        h = x
        x = self.norm1(x)
        x = self.attn(x, x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x, training)
        return x + h
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
        })
        return config

class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            "pos_emb",
            shape=[1, self.num_patches, self.embed_dim],
            initializer="random_normal",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim
        })
        return config


def build_vit(num_classes,
        image_size=32,
        patch_size=4,
        embed_dim=128,
        depth=8,
        num_heads=4,
        mlp_ratio=4.):
    num_patches = (image_size // patch_size) ** 2

    inputs = layers.Input((image_size, image_size, 3))

    # Patch embedding
    # x = PatchEmbedding(patch_size, embed_dim)(inputs)
    x = HierarchicalConvEmbedding(embed_dim=embed_dim)(inputs)
    # Positional embedding
    x = AddPositionEmbedding(num_patches, embed_dim)(x)

    # Transformer blocks
    for _ in range(depth):
        x = TransformerBlock(embed_dim, num_heads, mlp_ratio)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # x = layers.Flatten()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Regression head
    x = layers.Dense(embed_dim, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # 输出0~1

    model = tf.keras.Model(inputs, outputs)
    return model

class MLP(layers.Layer):
    def __init__(self, hidden_dim, out_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = float(dropout)

        self.fc1 = layers.Dense(hidden_dim, name="fc1")
        self.act = layers.Activation(tf.nn.gelu, name="gelu")
        #self.drop1 = layers.Dropout(dropout, name="drop1")
        self.fc2 = layers.Dense(out_dim, name="fc2")
        #self.drop2 = layers.Dropout(dropout, name="drop2")

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop1(x, training=training)
        x = self.fc2(x)
        #x = self.drop2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            #"dropout": self.dropout,
        })
        return config

class MixerBlock(layers.Layer):
    def __init__(self,
                 num_patches,
                 embed_dim,
                 token_mlp_dim,
                 channel_mlp_dim,
                 dropout=0.0,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim
        #self.dropout = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="ln_token")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="ln_channel")

        # token mixing
        self.token_mlp_1 = layers.Dense(token_mlp_dim, name="token_fc1")
        self.token_act = layers.Activation(tf.nn.gelu, name="token_gelu")
        #self.token_drop1 = layers.Dropout(dropout, name="token_drop1")
        self.token_mlp_2 = layers.Dense(num_patches, name="token_fc2")
        #self.token_drop2 = layers.Dropout(dropout, name="token_drop2")

        # channel mixing
        self.channel_mlp = MLP(
            hidden_dim=channel_mlp_dim,
            out_dim=embed_dim,
            dropout=dropout,
            name="channel_mlp"
        )

    def call(self, x, training=None):
        # Token mixing
        y = self.norm1(x)                  # (B, N, D)
        y = tf.transpose(y, [0, 2, 1])     # (B, D, N)
        y = self.token_mlp_1(y)            # (B, D, token_mlp_dim)
        y = self.token_act(y)
        #y = self.token_drop1(y, training=training)
        y = self.token_mlp_2(y)            # (B, D, N)
        #y = self.token_drop2(y, training=training)
        y = tf.transpose(y, [0, 2, 1])     # (B, N, D)
        x = x + y

        # Channel mixing
        y = self.norm2(x)
        y = self.channel_mlp(y, training=training)
        x = x + y

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
            "token_mlp_dim": self.token_mlp_dim,
            "channel_mlp_dim": self.channel_mlp_dim,
            #"dropout": self.dropout,
        })
        return config

def build_mlp_mixer_utkface_age(num_classes,
    image_size=32,
    embed_dim=128,
    channels=(64, 128),
    num_downsamples=2,
    use_bn=True,
    act="gelu",
    depth=8,
    token_mlp_dim=96,
    channel_mlp_ratio=3,
    dropout=0.1
):
    assert image_size % (2 ** num_downsamples) == 0

    out_size = image_size // (2 ** num_downsamples)
    num_patches = out_size * out_size
    channel_mlp_dim = int(embed_dim * channel_mlp_ratio)

    inputs = layers.Input(shape=(image_size, image_size, 3), name="input_image")

    x = HierarchicalConvEmbedding(
        image_size=image_size,
        embed_dim=embed_dim,
        channels=channels,
        num_downsamples=num_downsamples,
        use_bn=use_bn,
        act=act,
        name="hierarchical_conv_embedding"
    )(inputs)

    for i in range(depth):
        x = MixerBlock(
            num_patches=num_patches,
            embed_dim=embed_dim,
            token_mlp_dim=token_mlp_dim,
            channel_mlp_dim=channel_mlp_dim,
            dropout=dropout,
            name=f"mixer_block_{i}"
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(128, activation='relu', name="reg_dense_0")(x)
    #x = layers.Dropout(dropout, name="reg_drop_0")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="age_output")(x)

    model = tf.keras.Model(inputs, outputs, name="MLPMixer_UTKFace_Age_HConv")
    return model

