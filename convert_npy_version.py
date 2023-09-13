#In PyG 1.*
import torch
data = torch.load('data.pt')
torch.save(data.to_dict(), 'data_dict.pt')

# In PyG 2.*
import torch.utils.data
data_dict = torch.load('data_dict.pt')
data = data.from_dict(data_dict)
torch.save(data, 'data.pt')