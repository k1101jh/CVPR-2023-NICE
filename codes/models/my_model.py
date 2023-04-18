# https://github.com/rammyram/image_captioning

import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import SimpleViT


class SimpleEncoder(nn.Module):
    def __init__(self, embed_size):
        super(SimpleEncoder, self).__init__()
        # self.visual_transformer = SimpleViT(
        #     image_size=256,
        #     patch_size=32,
        #     num_classes=1000,
        #     dim=1024,
        #     depth=6,
        #     heads=16,
        #     mlp_dim=2048
        #     )
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
    
    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class SimpleDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(SimpleDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
    # def init_hidden(self, batch_size):
    #     return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
    #             torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        
    def forward(self, features, captions):
        captions = captions[:, :-1]
        self.batch_size = features.shape[0]
        # self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embedding(captions)
        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs
        
    def Predict(self, inputs, max_len=20):        
        final_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size) 
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.fc(lstm_out)  
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            final_output.append(max_idx.cpu().numpy()[0].item())             
            if (max_idx == 1 or len(final_output) >=20 ):
                break
            
            inputs = self.word_embedding(max_idx) 
            inputs = inputs.unsqueeze(1)             
        return final_output  
        




class MyModel(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        return x
