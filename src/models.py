import torch.nn as nn

import torchvision.models as models

class CifarResNet50(nn.Module):
    
    def __init__(self, model_config):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        
        self.num_classes = model_config['num_classes']
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, self.num_classes)
#         self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
#         print(x.shape)
        features = self.resnet(x).view(-1, 2048)
#         print(features.shape)
        out = self.fc(features)
#         out = self.softmax(out)
        return out

    

