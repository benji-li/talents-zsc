import torch.nn as nn
from ray.rllib.models.torch.misc import normc_initializer, SlimConv2d

def _create_convolutional_layers(in_channel, conv_filters, embedding_size):
    if isinstance(conv_filters, list):
        layers = []

        prev_out = in_channel

        for out_channel, kernel, stride, padding, activation in conv_filters:
            if padding == "same":
                pad = int((kernel-1)/2) # for 1 striding
            elif padding == "valid":
                pad = 0
            else:
                pad = 0
            active = activation
                
            if out_channel == "pool":
                layers.append(nn.MaxPool2d(kernel_size=kernel, stride=stride),)
            else:
                layers.append(SlimConv2d(prev_out, out_channel, kernel, stride, pad, activation_fn=active))
                prev_out = out_channel

        layers = nn.ModuleList(layers)
    else:
        return []

    return layers

def _create_dense_layers(sizes, layer_type = nn.Linear, activation_type = nn.ReLU, initializer = normc_initializer, activation_at_end=True):
    layers = []

    for idx, (in_size, out_size) in enumerate(sizes):
        layers.append(layer_type(in_size, out_size))

        if initializer is not None:
            initializer(layers[-1].weight)

        if activation_type is not None and (activation_at_end or idx < len(sizes)-1):
            layers.append(activation_type())

    layers = nn.ModuleList(layers)

    return layers

def _compute_layers(x, layers):
    if isinstance(layers, nn.ModuleList):
        for layer in layers:
            x = layer(x)
        return x
    elif isinstance(layers, nn.Sequential):
        return layers(x)
    else:
        raise ValueError("Unsupported layer container type")


def _preprocess_obs(actor_input_layers, input_dict):
    x = input_dict["obs"]["image"].float()
    # Reorder the state from (NHWC) to (NCHW).
    # Most images put channel last, but pytorch conv expects channel first.
    if isinstance(actor_input_layers, nn.ModuleList):
        if x.dim() < 4: # should be batch_size x channels x H x W
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
    else:
        if x.dim() < 2: # obs is vector
            x = x.unsqueeze(0)
    return x