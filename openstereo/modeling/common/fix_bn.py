def fix_bn(module):
    """Fix the batch normalization layers."""
    for module in module.modules():
        classname = module.__class__.__name__
        if classname.find('BatchNorm') != -1:
            module.eval()
    return module
