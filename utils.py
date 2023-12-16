import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

def xavier_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)

def he_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        init.zeros_(m.bias)

def prepare(model, initializer, device, num_classes):
    # Modify classifier
    if hasattr(model, 'classifier'):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise Exception("Unknown classifier layer type")
    
    # Initialize the weights
    model.apply(initializer)

    # Move to device
    model = model.to(device)

    # Set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    return model, optimizer