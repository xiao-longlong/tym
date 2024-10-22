import torch


# checkpoint_path = "/workspace/YOLOX_outputs/GTSRB_oril1loss_09211707/last_epoch_ckpt.pth"
checkpoint_path = "/workspace/YOLOX_outputs/GTSRB/GTSRB_100_09191032/epoch_5_ckpt.pth"
ckpt = torch.load(checkpoint_path, map_location="cpu")

if 'model' in ckpt:
    model_state_dict = ckpt['model']
for key in model_state_dict:
    print(key)

