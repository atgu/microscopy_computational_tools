import efficientnet.tfkeras as efn
import tensorflow as tf
import numpy as np
import glob

def cpcnn_model(pretrained_weights_path):
    # code extracted from https://github.com/cytomining/DeepProfiler/blob/master/plugins/models/efficientnet.py
    efn_model = efn.EfficientNetB0(input_shape=(128, 128, 5),
                                weights=None,
                                include_top=False)

    features = tf.compat.v1.keras.layers.GlobalAveragePooling2D(name="pool5")(efn_model.layers[-1].output)

    y = tf.compat.v1.keras.layers.Dense(490,
                                        activation="softmax",
                                        name="ClassProb")(features)
    model = tf.compat.v1.keras.models.Model(inputs=efn_model.input,
                                            outputs=[y])

    model.load_weights(pretrained_weights_path)

    # code from https://github.com/cytomining/DeepProfiler/blob/master/deepprofiler/learning/profiling.py
    feat_extractor = tf.compat.v1.keras.Model(
        model.inputs, 
        model.get_layer("block6a_activation").output)

    def eval_network(x):
        # change x.shape from [num_images, num_channels, 128, 128] into [num_images, 128, 128, num_channels]
        x = x.numpy()
        x = np.moveaxis(x, 1, 3)
        x = tf.convert_to_tensor(x)

        # code from https://github.com/cytomining/DeepProfiler/blob/master/deepprofiler/learning/profiling.py
        features = feat_extractor.predict([x], verbose=0)
        while len(features.shape) > 2:
            features = np.mean(features, axis=1)
        return features
    
    return eval_network