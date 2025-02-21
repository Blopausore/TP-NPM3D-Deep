import torch.nn as nn


def main():
        # loss and network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    input = ME.SparseTensor(feat, coordinates=coords)
    # Forward
    output = net(input)

    # Loss
    loss = criterion(output.F, label)