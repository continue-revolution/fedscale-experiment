import torch
import torchvision
mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
resnet18 = torchvision.models.resnet18(pretrained=True)
shufflenetV2 = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
models = [mobilenetv2, resnet18, shufflenetV2]
model_size = [0, 0, 0]
for i, model in enumerate(models):
        for _, param in enumerate(model.parameters()):
                    model_size[i] += param.numel() * 4
print(model_size)


import time
import matplotlib.pyplot as plt
import random
num_iteration = 100

model_name = ['MobileNetV2', 'ResNet18', 'ShuffleNetV2']
fig, ax = plt.subplots()
with torch.no_grad():
    for idx, model in enumerate(models):
        for thread_num in [1, 2, 4, 16, 32]:
            torch.set_num_threads(thread_num)
            for device in ['cpu', 'cuda']:
                model = model.to(device)
                if idx == 0:
                    _model = torchvision.models.mobilenet_v2(pretrained=True).to(device)
                elif idx == 1:
                    _model = torchvision.models.resnet18(pretrained=True).to(device)
                else:
                    _model = torchvision.models.shufflenet_v2_x1_0(pretrained=True).to(device)
                time_list = []
                size_list = []
                print(f'{model_name[idx]}-{device}-{thread_num}')
                start = time.time()
                for i in range(num_iteration):
                    for key in model.state_dict():
                        rand = torch.tensor(random.random(), dtype=model.state_dict()[key].data.dtype)
                        _model.state_dict()[key].data += rand * model.state_dict()[key].data
                    time_diff = time.time() - start
                    size_list.append((i + 1) * model_size[idx])
                    time_list.append(time_diff)
                print(time_list[-1])
                ax.plot(size_list, time_list,label=f'{model_name[idx]}-{device}-{thread_num}')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel('aggregation size / (model_size * num_iteration)')
ax.set_ylabel('duration / s')
ax.set_title('Duration - Aggregation Size Plot')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.savefig(f"{time.time()}.png", bbox_inches="tight")