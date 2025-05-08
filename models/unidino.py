import torch
import models.archs.vision_transformer as vits
import numpy as np

# based on https://github.com/Bayer-Group/uniDINO/blob/main/inference.py

def unidino_model(pretrained_weights_path, embed_dim=384):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    patch_size = 16
    checkpoint_key = "teacher"
    arch = "vit_small"

    model = vits.__dict__[arch](
        patch_size=patch_size,
        drop_path_rate=0.1,
        in_chans=1,
        embed_dim=embed_dim,
        cls_reduced_dim=None,
    )

    model.to(device)

    state_dict = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
    
    assert "teacher" in state_dict, "teacher not in state dict"
    teacher = state_dict["teacher"]
    teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
    teacher = {
        k.replace("backbone.", ""): v for k, v in teacher.items()
    }
    msg = model.load_state_dict(teacher, strict=False)
    
    
    model.eval()

    tmp = torch.tensor(np.random.uniform(0, 1, (56, 1, 224, 224)).astype(np.float32)).to(device)
    with torch.inference_mode():
        assert model(tmp).shape == (56, embed_dim)

    def eval_network(x, _):
        crop_size = 224
        stride = crop_size

        num_channels = x.shape[0]

        # create 224 x 224 crops
        x = x.unfold(dimension=1, size=crop_size, step=stride).unfold(
            dimension=2, size=crop_size, step=stride
        )

        # dimensio of x is num_channels x num_crops_x x num_crops_y x 224 x 224
        # reshape to model input dimension (num_channels * num_crops) x 1 x 224 x 224
        x = x.reshape(-1, 1, crop_size, crop_size)

        with torch.inference_mode():
            x = torch.as_tensor(x)
            y = model(x.to(device)).cpu().numpy()

        # model output dimension is (num_channels * num_crops) x 384
        # reshape to num_crops x (num_channels * 384)
        y = y.reshape(num_channels, -1, embed_dim)
        y = np.swapaxes(y, 0, 1)
        y = y.reshape(-1, num_channels * embed_dim)

        # compute mean over crops
        y = np.mean(y, axis=0)

        return [y.tolist()]

    return eval_network