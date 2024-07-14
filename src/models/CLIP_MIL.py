import torch
import random
import torch.nn as nn
import numpy as np
from src.models.utils import *

from src import clip


class PromptEmbedding(nn.Module):
    def __init__(
            self,
            token_embedding,
            context_length: int,
            template: str,
            use_CoOP: bool,
            device: str,
    ):
        super(PromptEmbedding, self).__init__()
        """

        """

        self.device = device
        self.context_length = context_length
        self.tokenizer = clip.tokenize
        self.token_embedding = token_embedding
        self.use_CoOP = use_CoOP
        self.template = template

        if self.use_CoOP:
            prompts = [template]
            prompts = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompts)

            n_ctx = prompts.argmax(dim=1)[0]
            print(f"length of template: {n_ctx}")
            self.register_buffer("token_prefix", embedding[0, :1, :])  # SOS
            self.ctx_embedding = nn.Parameter(embedding[0, 1: n_ctx, :]) # template
            self.suffix_length = context_length - n_ctx

    def forward_fixed(self, descriptions: dict):
        prompts = []
        for class_name, class_descriptions in descriptions.items():
            prompts.append(f"{self.template} {class_name}, which is {random.choice(class_descriptions)}")
        prompts = self.tokenizer(prompts, context_length=self.context_length).to(self.device)

        with torch.no_grad():
            embedding = self.token_embedding(prompts)
        eos_position = prompts.argmax(dim=-1)
        return embedding, eos_position

    def forward_CoOP(self, descriptions: dict):
        n_classes = len(descriptions)
        ctx = self.ctx_embedding
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_classes, -1, -1)
        
        prefix = self.token_prefix
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(0).expand(n_classes, -1, -1)

        prompts = []
        for class_name, class_description in descriptions.items():
            prompts.append(f"{class_name}, which is {random.choice(class_description)}")
        prompts = self.tokenizer(prompts, context_length=self.suffix_length).to(self.device)
        with torch.no_grad():
            suffix = self.token_embedding(prompts)

        embeddings = torch.cat([
            prefix,
            ctx,
            suffix
        ], dim=1)
        eos_postion = prompts.argmax(dim=-1) + self.context_length - self.suffix_length
        return embeddings, eos_postion

    def forward(self, descriptions: dict):
        if self.use_CoOP:
            return self.forward_CoOP(descriptions)
        else:
            return self.forward_fixed(descriptions)


class CLIP_MIL(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            text_enc_name: str,
            pooling_method: str,
            template: str,
            bag_descriptions: dict,
            use_CoOP: bool,
            device: str,
            **kwargs
    ):
        super(CLIP_MIL, self).__init__()
        self.feat_dim = feat_dim
        self.device = device if torch.cuda.is_available() else "cpu"
        self.text_encoder = clip.load_text_encoder(name=text_enc_name, device=self.device)
        self.pooling_method = pooling_method
        self.bag_descriptions = bag_descriptions

        # set pooling method
        if pooling_method == "mean":
            self.instance_pooling = MeanPooling()
        elif pooling_method == "transmil":
            self.instance_pooling = TransMILPooling(feat_dim=feat_dim, hidden_dim=feat_dim)
        elif self.pooling_method == "instance":
            self.instance_descriptions = kwargs["instance_descriptions"]
            self.instance_pooling = CrossAttnBlock(dim=feat_dim, num_heads=4)
        elif self.pooling_method == "instance_stain":
            self.instance_descriptions = kwargs["instance_descriptions"]
            self.instance_pooling = CrossAttnBlock(dim=feat_dim, num_heads=4)
            self.stain_descriptions = kwargs["stain_descriptions"]
            self.stain_pooling = CrossAttnBlock(dim=feat_dim, num_heads=4)
            self.fc = nn.Linear(in_features=2*feat_dim, out_features=feat_dim)
        else:
            raise ValueError("Unknown pooling method: {}".format(pooling_method))

        self.prompt_embedding = PromptEmbedding(
            token_embedding=self.text_encoder.token_embedding,
            context_length=77,
            template=template,
            use_CoOP=use_CoOP,
            device=self.device,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _apply_consine_similarity(self, image_features, text_features, scale):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits = scale * image_features @ text_features.t()
        return logits

    def forward(self, image_features, return_attn=False):
        # TODO: if validation, ensemble the descriptions
        # image_features: [1, N, dim]
        output_dict = {}
        bag_prompts, bag_eos_position = self.prompt_embedding(self.bag_descriptions)
        bag_prompts = self.text_encoder(bag_prompts, bag_eos_position)

        if self.pooling_method == "mean":
            wsi_feature = self.instance_pooling(image_features)
        elif self.pooling_method == "transmil":
            wsi_feature, instance_attn = self.instance_pooling(image_features, return_attn)
            output_dict["attn"] = instance_attn
        elif self.pooling_method == "instance":
            cls_prompts, cls_eos_position = self.prompt_embedding(self.instance_descriptions)
            cls_prompts = self.text_encoder(cls_prompts, cls_eos_position).unsqueeze(0)
            image_features, attn = self.instance_pooling(cls_prompts, image_features)
            wsi_feature = torch.mean(image_features.squeeze(0), dim=0, keepdim=True)
            output_dict["attn"] = attn
        elif self.pooling_method == "instance_stain":
            cls_prompts, cls_eos_position = self.prompt_embedding(self.instance_descriptions)
            cls_prompts = self.text_encoder(cls_prompts, cls_eos_position).unsqueeze(0)
            cls_features, cls_attn = self.instance_pooling(cls_prompts, image_features)

            stain_prompts, stain_eos_position = self.prompt_embedding(self.stain_descriptions)
            stain_prompts = self.text_encoder(stain_prompts, stain_eos_position).unsqueeze(0)
            stain_features, stain_attn = self.stain_pooling(stain_prompts, image_features)

            wsi_feature = torch.cat([
                torch.mean(cls_features.squeeze(0), dim=0, keepdim=True),
                torch.mean(stain_features.squeeze(0), dim=0, keepdim=True)
            ], dim=-1)
            wsi_feature = self.fc(wsi_feature)
            output_dict["cls_attn"] = cls_attn
            output_dict["stain_attn"] = stain_attn
        else:
            raise ValueError("Unknown pooling method: {}".format(self.pooling_method))

        logit_scale = self.logit_scale.exp()
        logits = self._apply_consine_similarity(wsi_feature, bag_prompts, logit_scale) # [1, classes]
        output_dict["logits"] = logits
        return output_dict

if __name__ == '__main__':
    import yaml
    with open("/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_transmilpooling.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    model = CLIP_MIL(
        **config["model"]
    )
    output = model(torch.randn(1, 1024, 512))
    for k, v in output.items():
        if v is not None:
            print(k, v.shape)
        else:
            print(k)
