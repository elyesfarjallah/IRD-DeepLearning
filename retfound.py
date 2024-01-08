
import torch
import RETFound_MAE.models_vit as models_vit
from RETFound_MAE.util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

# call the model
def retfound(pretrained: bool = True, weights : str = './RETFound_MAE/RETFound_cfp_weights.pth'):
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=2,
        drop_path_rate=0.2,
        global_pool=True,
    )

    # load RETFound weights
    checkpoint = torch.load(weights, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    return model