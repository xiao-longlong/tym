import torch
import torch.nn as nn

def remove_unnecessary_layers(checkpoint_path, save_path):
    # 加载 checkpoint 文件
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # # 输出原始参数键
    # print("Original keys in checkpoint:")
    # for key in ckpt['state_dict'].keys():
    #     print(key)

    if 'model' in ckpt:
        print("right")
    #     # model_state_dict = ckpt['model']
    # else:
    #     model_state_dict = ckpt  # 如果没有 state_dict，直接使用 ckpt

    # 指定要删除的层的键
    keys_to_remove = [
        "head.ClsAttentionLayers.0.0.conv.weight", 
        "head.ClsAttentionLayers.0.0.bn.weight", 
        "head.ClsAttentionLayers.0.0.bn.bias", 
        "head.ClsAttentionLayers.0.0.bn.running_mean", 
        "head.ClsAttentionLayers.0.0.bn.running_var", 
        "head.ClsAttentionLayers.0.0.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.0.1.conv.weight", 
        "head.ClsAttentionLayers.0.1.bn.weight", 
        "head.ClsAttentionLayers.0.1.bn.bias", 
        "head.ClsAttentionLayers.0.1.bn.running_mean", 
        "head.ClsAttentionLayers.0.1.bn.running_var", 
        "head.ClsAttentionLayers.0.1.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.0.2.conv.weight", 
        "head.ClsAttentionLayers.0.2.bn.weight", 
        "head.ClsAttentionLayers.0.2.bn.bias", 
        "head.ClsAttentionLayers.0.2.bn.running_mean", 
        "head.ClsAttentionLayers.0.2.bn.running_var", 
        "head.ClsAttentionLayers.0.2.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.0.3.conv.weight", 
        "head.ClsAttentionLayers.0.3.bn.weight", 
        "head.ClsAttentionLayers.0.3.bn.bias", 
        "head.ClsAttentionLayers.0.3.bn.running_mean", 
        "head.ClsAttentionLayers.0.3.bn.running_var", 
        "head.ClsAttentionLayers.0.3.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.1.0.conv.weight", 
        "head.ClsAttentionLayers.1.0.bn.weight", 
        "head.ClsAttentionLayers.1.0.bn.bias", 
        "head.ClsAttentionLayers.1.0.bn.running_mean", 
        "head.ClsAttentionLayers.1.0.bn.running_var", 
        "head.ClsAttentionLayers.1.0.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.1.1.conv.weight", 
        "head.ClsAttentionLayers.1.1.bn.weight", 
        "head.ClsAttentionLayers.1.1.bn.bias", 
        "head.ClsAttentionLayers.1.1.bn.running_mean", 
        "head.ClsAttentionLayers.1.1.bn.running_var", 
        "head.ClsAttentionLayers.1.1.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.1.2.conv.weight", 
        "head.ClsAttentionLayers.1.2.bn.weight", 
        "head.ClsAttentionLayers.1.2.bn.bias", 
        "head.ClsAttentionLayers.1.2.bn.running_mean", 
        "head.ClsAttentionLayers.1.2.bn.running_var", 
        "head.ClsAttentionLayers.1.2.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.1.3.conv.weight", 
        "head.ClsAttentionLayers.1.3.bn.weight", 
        "head.ClsAttentionLayers.1.3.bn.bias", 
        "head.ClsAttentionLayers.1.3.bn.running_mean", 
        "head.ClsAttentionLayers.1.3.bn.running_var", 
        "head.ClsAttentionLayers.1.3.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.2.0.conv.weight", 
        "head.ClsAttentionLayers.2.0.bn.weight", 
        "head.ClsAttentionLayers.2.0.bn.bias", 
        "head.ClsAttentionLayers.2.0.bn.running_mean", 
        "head.ClsAttentionLayers.2.0.bn.running_var", 
        "head.ClsAttentionLayers.2.0.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.2.1.conv.weight", 
        "head.ClsAttentionLayers.2.1.bn.weight", 
        "head.ClsAttentionLayers.2.1.bn.bias", 
        "head.ClsAttentionLayers.2.1.bn.running_mean", 
        "head.ClsAttentionLayers.2.1.bn.running_var", 
        "head.ClsAttentionLayers.2.1.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.2.2.conv.weight", 
        "head.ClsAttentionLayers.2.2.bn.weight", 
        "head.ClsAttentionLayers.2.2.bn.bias", 
        "head.ClsAttentionLayers.2.2.bn.running_mean", 
        "head.ClsAttentionLayers.2.2.bn.running_var", 
        "head.ClsAttentionLayers.2.2.bn.num_batches_tracked", 
        "head.ClsAttentionLayers.2.3.conv.weight", 
        "head.ClsAttentionLayers.2.3.bn.weight", 
        "head.ClsAttentionLayers.2.3.bn.bias", 
        "head.ClsAttentionLayers.2.3.bn.running_mean", 
        "head.ClsAttentionLayers.2.3.bn.running_var", 
        "head.ClsAttentionLayers.2.3.bn.num_batches_tracked", 
        "head.RegAttentionLayers.0.conv.weight", 
        "head.RegAttentionLayers.0.bn.weight", 
        "head.RegAttentionLayers.0.bn.bias", 
        "head.RegAttentionLayers.0.bn.running_mean", 
        "head.RegAttentionLayers.0.bn.running_var", 
        "head.RegAttentionLayers.0.bn.num_batches_tracked", 
        "head.RegAttentionLayers.1.conv.weight", 
        "head.RegAttentionLayers.1.bn.weight", 
        "head.RegAttentionLayers.1.bn.bias", 
        "head.RegAttentionLayers.1.bn.running_mean", 
        "head.RegAttentionLayers.1.bn.running_var", 
        "head.RegAttentionLayers.1.bn.num_batches_tracked", 
        "head.RegAttentionLayers.2.conv.weight", 
        "head.RegAttentionLayers.2.bn.weight", 
        "head.RegAttentionLayers.2.bn.bias", 
        "head.RegAttentionLayers.2.bn.running_mean", 
        "head.RegAttentionLayers.2.bn.running_var", 
        "head.RegAttentionLayers.2.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.0.0.conv.weight", 
        "head.ObjAttentionLayers.0.0.bn.weight", 
        "head.ObjAttentionLayers.0.0.bn.bias", 
        "head.ObjAttentionLayers.0.0.bn.running_mean", 
        "head.ObjAttentionLayers.0.0.bn.running_var", 
        "head.ObjAttentionLayers.0.0.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.0.1.conv.weight", 
        "head.ObjAttentionLayers.0.1.bn.weight", 
        "head.ObjAttentionLayers.0.1.bn.bias", 
        "head.ObjAttentionLayers.0.1.bn.running_mean", 
        "head.ObjAttentionLayers.0.1.bn.running_var", 
        "head.ObjAttentionLayers.0.1.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.0.2.conv.weight", 
        "head.ObjAttentionLayers.0.2.bn.weight", 
        "head.ObjAttentionLayers.0.2.bn.bias", 
        "head.ObjAttentionLayers.0.2.bn.running_mean", 
        "head.ObjAttentionLayers.0.2.bn.running_var", 
        "head.ObjAttentionLayers.0.2.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.1.0.conv.weight", 
        "head.ObjAttentionLayers.1.0.bn.weight", 
        "head.ObjAttentionLayers.1.0.bn.bias", 
        "head.ObjAttentionLayers.1.0.bn.running_mean", 
        "head.ObjAttentionLayers.1.0.bn.running_var", 
        "head.ObjAttentionLayers.1.0.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.1.1.conv.weight", 
        "head.ObjAttentionLayers.1.1.bn.weight", 
        "head.ObjAttentionLayers.1.1.bn.bias", 
        "head.ObjAttentionLayers.1.1.bn.running_mean", 
        "head.ObjAttentionLayers.1.1.bn.running_var", 
        "head.ObjAttentionLayers.1.1.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.1.2.conv.weight", 
        "head.ObjAttentionLayers.1.2.bn.weight", 
        "head.ObjAttentionLayers.1.2.bn.bias", 
        "head.ObjAttentionLayers.1.2.bn.running_mean", 
        "head.ObjAttentionLayers.1.2.bn.running_var", 
        "head.ObjAttentionLayers.1.2.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.2.0.conv.weight", 
        "head.ObjAttentionLayers.2.0.bn.weight", 
        "head.ObjAttentionLayers.2.0.bn.bias", 
        "head.ObjAttentionLayers.2.0.bn.running_mean", 
        "head.ObjAttentionLayers.2.0.bn.running_var", 
        "head.ObjAttentionLayers.2.0.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.2.1.conv.weight", 
        "head.ObjAttentionLayers.2.1.bn.weight", 
        "head.ObjAttentionLayers.2.1.bn.bias", 
        "head.ObjAttentionLayers.2.1.bn.running_mean", 
        "head.ObjAttentionLayers.2.1.bn.running_var", 
        "head.ObjAttentionLayers.2.1.bn.num_batches_tracked", 
        "head.ObjAttentionLayers.2.2.conv.weight", 
        "head.ObjAttentionLayers.2.2.bn.weight", 
        "head.ObjAttentionLayers.2.2.bn.bias", 
        "head.ObjAttentionLayers.2.2.bn.running_mean", 
        "head.ObjAttentionLayers.2.2.bn.running_var", 
        "head.ObjAttentionLayers.2.2.bn.num_batches_tracked", 
        "head.cls_preds1.0.weight", 
        "head.cls_preds1.0.bias", 
        "head.cls_preds1.1.weight", 
        "head.cls_preds1.1.bias", 
        "head.cls_preds1.2.weight", 
        "head.cls_preds1.2.bias", 
        "head.reg_preds1.0.weight", 
        "head.reg_preds1.0.bias", 
        "head.reg_preds1.1.weight", 
        "head.reg_preds1.1.bias", 
        "head.reg_preds1.2.weight", 
        "head.reg_preds1.2.bias", 
        "head.obj_preds1.0.weight", 
        "head.obj_preds1.0.bias", 
        "head.obj_preds1.1.weight", 
        "head.obj_preds1.1.bias", 
        "head.obj_preds1.2.weight", 
        "head.obj_preds1.2.bias"
    ]
    

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    optiflag_del0 = []
    optiflag_del1 = []
    optiflag_del2 = []


    for k in ckpt['model']:
        if ".bias" in k:
            pg2.append(k)
            if not (k in keys_to_remove):
                optiflag_del2.append(True)

        if ".bn.weight" in k:
            pg0.append(k)
            if not (k in keys_to_remove):
                optiflag_del0.append(True)

        elif ".weight" in k:
            pg1.append(k)
            if not (k in keys_to_remove):
                optiflag_del1.append(True)

    for new_key, old_key in enumerate(list(ckpt['optimizer']['state'].keys())):
        ckpt['optimizer']['state'][new_key] = ckpt['optimizer']['state'].pop(old_key)


    ckpt['optimizer']['param_groups'][0]['params'] = [i for i in range(0,len(optiflag_del0))]
    ckpt['optimizer']['param_groups'][1]['params'] = [i for i in range(len(optiflag_del0),len(optiflag_del0+optiflag_del1))]
    ckpt['optimizer']['param_groups'][2]['params'] = [i for i in range(len(optiflag_del0+optiflag_del1),len(optiflag_del0+optiflag_del1+optiflag_del2))]


# 删除不必要的层
    for key in keys_to_remove:
        if key in ckpt['model']:
            del ckpt['model'][key]
            print(f"Removed: {key}")
        else:
            print(f"Key not found in checkpoint: {key}")
    
    # 保存修改后的 checkpoint 文件
    torch.save(ckpt, save_path)
    print(f"Modified checkpoint saved to: {save_path}")

if __name__ == "__main__":
    # 指定要加载的 checkpoint 路径和要保存的路径
    checkpoint_path = "/workspace/YOLOX_outputs/GTSRB/GTSRB_ori_09170122_4090/best_ckpt.pth"  # 替换为你的 checkpoint 路径
    save_path = "/workspace/YOLOX_outputs/GTSRB/GTSRB_2201_09191033/resume.pth"  # 替换为你想保存的路径
    
    # 调用函数
    remove_unnecessary_layers(checkpoint_path, save_path)
