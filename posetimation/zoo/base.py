
import torch.nn as nn




class BaseModel(nn.Module):

    def forward(self, *input):
        raise NotImplementedError

    def init_weights(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_model_hyper_parameters(cls, cfg):
        raise NotImplementedError
