import torch

# 从 torch.hub 加载预训练 carvana 模型
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

# predict.py 期望 load_state_dict(state_dict)，并且会 pop('mask_values')
# 所以我们把 mask_values 一起存进去，保证 predict.py 不报错、输出正确
state = net.state_dict()
state['mask_values'] = [0, 1]

torch.save(state, 'carvana_pretrained.pth')
print("Saved to carvana_pretrained.pth")
