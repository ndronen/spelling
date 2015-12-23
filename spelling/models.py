from keras.models import (Sequential, Graph)
from keras.layers.core import (Flatten)

from modeling.builders import (build_embedding_layer,
        build_convolutional_layer, build_pooling_layer, build_dense_layer,
        load_weights, build_optimizer)
from modeling.utils import ModelConfig

def build_dictionary_model_args(input_width, n_classes, optimizer, **kwargs):
    args = {
            "train_embeddings": True,
            "embedding_weights": None,
            "regularization_layer": "",
            "dropout_p": 0.5,
            "dropout_p_conv": 0.0,
            "n_embeddings": 100,
            "n_embed_dims": 50,
            #"n_residual_blocks": None,
            #"n_layers_per_residual_block": 2,
            "n_fully_connected": None,
            "border_mode": "valid",
            "n_filters": 500,
            "n_hidden": 500,
            "filter_width": 4,
            "loss": "categorical_crossentropy",
            "patience": 20,
            "batch_size": 128,
            "embedding_max_norm": 1000,
            "filter_max_norm": 1000,
            "dense_max_norm": 1000,
            "l2_penalty": 0.0000,
            "clipnorm": 0
        }

    args['input_width'] = input_width
    args['n_classes'] = n_classes
    args['optimizer'] = optimizer

    for k,v in kwargs.items():
        args[k] = v

    return ModelConfig(**args)

def build_dictionary_model(args):
    model = Sequential()
    model.add(build_embedding_layer(args))
    model.add(build_convolutional_layer(args))
    model.add(build_pooling_layer(args))
    model.add(Flatten())
    model.add(build_dense_layer(args, args.n_classes, activation='softmax'))

    load_weights(args, model)

    optimizer = build_optimizer(args)
    model.compile(loss=args.loss, optimizer=optimizer)

    return model
