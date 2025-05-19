def count_model_params(model, should_log = True):
    """ Counting the number of learnable parameters in a nn.Module """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if should_log:
        print(f"Learnable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
    
    return (total_params, trainable_params)
