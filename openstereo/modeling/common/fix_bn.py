def fix_bn(module):
    """Fix the batch normalization layers."""
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    return module
