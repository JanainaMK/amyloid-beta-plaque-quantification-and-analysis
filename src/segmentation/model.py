import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def unet(model_file=''):
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'unet',
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model.double()
    model.to(DEVICE)
    if model_file != '':
        model.load_state_dict(torch.load(model_file))
    return model


def get_model_file_path(name: str, version: str):
    return f'models/unet/{name}-{version}.pt'


def get_model_name(name, model_type, downsample_lvl, batch_size, patch_size, learning_rate):
    return f'{name}-{model_type}-{downsample_lvl}x-bs{batch_size}-ps{patch_size}-lr{learning_rate}'


def save_model_from_path(model, path):
    torch.save(model.state_dict(), path)


def save_model_from_name(model, name, version):
    path = get_model_file_path(name, version)
    save_model_from_path(model, path)

