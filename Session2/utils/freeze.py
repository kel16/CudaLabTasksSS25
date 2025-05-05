def freezeParams(model):
    """ freezes the weights for all of the network """
    for param in model.parameters():
        param.requires_grad = False

def unfreezeParams(model):
    """ unfreezes the weights for all of the network """
    for param in model.parameters():
        param.requires_grad = True

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}, trainable: {trainable_params:,}")
