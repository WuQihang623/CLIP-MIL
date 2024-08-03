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

    def forward_fixed(self, descriptions: dict, ensemble=False):
        if ensemble:
            embeddings = {}
            eos_postions = {}
            for cls_name, cls_descriptions in descriptions.items():
                prompts = []
                prompts.extend([f"{self.template} {cls_name}, which is {random.choice(description)}"
                                 for description in cls_descriptions])
                prompts = self.tokenizer(prompts, context_length=self.context_length).to(self.device)
                with torch.no_grad():
                    embedding = self.token_embedding(prompts)
                eos_postion = prompts.argmax(dim=-1)
                embeddings[cls_name] = embedding
                eos_postions[cls_name] = eos_postion
            return embeddings, eos_postions
        else:
            prompts = []
            for class_name, class_descriptions in descriptions.items():
                prompts.append(f"{self.template} {class_name}, which is {random.choice(class_descriptions)}")
            prompts = self.tokenizer(prompts, context_length=self.context_length).to(self.device)

            with torch.no_grad():
                embedding = self.token_embedding(prompts)
            eos_position = prompts.argmax(dim=-1)
            return embedding, eos_position

    def forward_CoOP(self, descriptions: dict, ensemble=False):
        if ensemble:
            embeddings = {}
            eos_postions = {}
            for cls_name, cls_descriptions in descriptions.items():
                ctx = self.ctx_embedding
                prefix = self.token_prefix

                n = len(cls_descriptions)
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(n, -1, -1)
                if prefix.dim() == 2:
                    prefix = prefix.unsqueeze(0).expand(n, -1, -1)

                prompts = []
                for description in cls_descriptions:
                    prompts.append(f"{cls_name}, which is {description}")
                prompts = self.tokenizer(prompts, context_length=self.suffix_length).to(self.device)
                with torch.no_grad():
                    suffix = self.token_embedding(prompts)

                embedding = torch.cat([
                    prefix,
                    ctx,
                    suffix
                ], dim=1)
                eos_postion = prompts.argmax(dim=-1) + self.context_length - self.suffix_length
                embeddings[cls_name] = embedding
                eos_postions[cls_name] = eos_postion
            return embeddings, eos_postions

        else:
            n_classes = len(descriptions)
            ctx = self.ctx_embedding
            prefix = self.token_prefix
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(n_classes, -1, -1)

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

    def forward(self, descriptions: dict, ensemble=False):
        if self.use_CoOP:
            return self.forward_CoOP(descriptions, ensemble)
        else:
            return self.forward_fixed(descriptions, ensemble)


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
            ensemble: bool = False,
            attention: str = "cross",
            num_head: int = 1,
            sample_method: str = "None",
            visual_token: int = 0,
            **kwargs
    ):
        super(CLIP_MIL, self).__init__()
        self.feat_dim = feat_dim
        self.device = device if torch.cuda.is_available() else "cpu"
        self.text_encoder = clip.load_text_encoder(name=text_enc_name, device=self.device)
        self.pooling_method = pooling_method
        self.bag_descriptions = bag_descriptions
        self.ensemble = ensemble
        self.sample_method = sample_method
        if self.sample_method != "None":
            self.sample_ratio_min = kwargs["sample_ratio_min"]
            self.sample_ratio_max = kwargs["sample_ratio_max"]
        self.visual_tokens = None
        if visual_token > 0:
            self.visual_tokens = nn.Parameter(torch.zeros(1, visual_token, feat_dim))
            self.norm_tokens = nn.LayerNorm(feat_dim)
            nn.init.trunc_normal_(self.visual_tokens, std=.02)

        # set pooling method
        if pooling_method == "mean":
            self.instance_pooling = MeanPooling()
        elif pooling_method == "transmil":
            self.instance_pooling = TransMILPooling(feat_dim=feat_dim, hidden_dim=feat_dim)
        elif pooling_method == "abmil":
            self.instance_pooling = Attn_Net_Gated(L=feat_dim, D=feat_dim)
        elif self.pooling_method == "instance":
            self.instance_descriptions = kwargs["instance_descriptions"]
            self.instance_pooling = CrossAttnBlock(dim=feat_dim, num_heads=num_head, attention=attention)
        elif self.pooling_method == "stain":
            self.stain_descriptions = kwargs["stain_descriptions"]
            self.stain_pooling = CrossAttnBlock(dim=feat_dim, num_heads=num_head, attention=attention)
        elif self.pooling_method == "instance_stain":
            self.instance_descriptions = kwargs["instance_descriptions"]
            self.stain_descriptions = kwargs["stain_descriptions"]
            self.fusion = kwargs.get("fusion", "concat")
            self.instance_pooling = CrossAttnBlock(dim=feat_dim, num_heads=num_head, attention=attention)
            self.stain_pooling = CrossAttnBlock(dim=feat_dim, num_heads=num_head, attention=attention)
            if self.fusion == "concat":
                self.fc = nn.Linear(in_features=2*feat_dim, out_features=feat_dim)
            elif self.fusion == "group":
                self.fusion_block = GroupingBlock(dim=feat_dim, num_heads=1, num_group_token=4, num_output_group=1)
            else:
                raise ValueError("Unknown fusion method: {}".format(self.fusion))
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

    def _text_encoder(self, descriptions: dict, ensemble=False):
        if ensemble:
            ensemble_prompts = []
            prompts, eos_positions = self.prompt_embedding(descriptions, ensemble)
            for cls_name in descriptions.keys():
                embedding = self.text_encoder(prompts[cls_name], eos_positions[cls_name])
                ensemble_prompts.append(torch.mean(embedding, dim=0, keepdim=True))
            ensemble_prompts = torch.cat(ensemble_prompts, dim=0)
            return ensemble_prompts
        else:
            prompts, eos_positions = self.prompt_embedding(descriptions, ensemble)
            embedding = self.text_encoder(prompts, eos_positions)
            return embedding

    def sample(self, image_features):
        B, N, D = image_features.shape
        sample_ratio = random.uniform(self.sample_ratio_min, self.sample_ratio_max)
        k = int(sample_ratio * N)
        indices = torch.rand(B, N).argsort(dim=1)[:, :k]
        batch_indices = torch.arange(B).unsqueeze(1).repeat(1, k)
        image_features = image_features[batch_indices, indices, :]
        return image_features

    def forward(self, image_features, return_attn=False):
        # image_features: [1, N, dim]
        output_dict = {}

        ensemble = self.ensemble

        bag_prompts = self._text_encoder(self.bag_descriptions, ensemble)

        if self.training and self.sample_method != "None":
            image_features = self.sample(image_features)

        if self.visual_tokens is not None:



        if self.pooling_method == "mean":
            wsi_feature = self.instance_pooling(image_features)
        elif self.pooling_method == "transmil":
            wsi_feature, instance_attn = self.instance_pooling(image_features, return_attn)
            output_dict["inst_attn"] = instance_attn
        elif self.pooling_method == "abmil":
            wsi_feature, instance_attn = self.instance_pooling(image_features)
            output_dict["inst_attn"] = instance_attn
        elif self.pooling_method == "instance":
            cls_prompts = self._text_encoder(self.instance_descriptions, ensemble).unsqueeze(0)
            image_features, attn = self.instance_pooling(cls_prompts, image_features)
            wsi_feature = torch.mean(image_features.squeeze(0), dim=0, keepdim=True)
            output_dict["inst_attn"] = attn
        elif self.pooling_method == "stain":
            stain_prompts = self._text_encoder(self.stain_descriptions, ensemble).unsqueeze(0)
            image_features, attn = self.stain_pooling(stain_prompts, image_features)
            wsi_feature = torch.mean(image_features.squeeze(0), dim=0, keepdim=True)
            output_dict["stain_attn"] = attn
        elif self.pooling_method == "instance_stain":
            cls_prompts = self._text_encoder(self.instance_descriptions, ensemble).unsqueeze(0)
            cls_features, cls_attn = self.instance_pooling(cls_prompts, image_features)

            stain_prompts = self._text_encoder(self.stain_descriptions, ensemble).unsqueeze(0)
            stain_features, stain_attn = self.stain_pooling(stain_prompts, image_features)

            if self.fusion == "concat":
                wsi_feature = torch.cat([
                    torch.mean(cls_features.squeeze(0), dim=0, keepdim=True),
                    torch.mean(stain_features.squeeze(0), dim=0, keepdim=True)
                ], dim=-1)
                wsi_feature = self.fc(wsi_feature)
            elif self.fusion == "group":
                wsi_feature = torch.cat([
                    cls_features, stain_features
                ], dim=1)
                wsi_feature, fusion_attn_dict = self.fusion_block(wsi_feature)
                wsi_feature = torch.mean(wsi_feature.squeeze(0), dim=0, keepdim=True)
            else:
                raise ValueError("Unknown fusion method: {}".format(self.fusion))
            output_dict["inst_attn"] = cls_attn
            output_dict["stain_attn"] = stain_attn
        else:
            raise ValueError("Unknown pooling method: {}".format(self.pooling_method))

        logit_scale = self.logit_scale.exp()
        logits = self._apply_consine_similarity(wsi_feature, bag_prompts, logit_scale) # [1, classes]
        output_dict["logits"] = logits
        output_dict["features"] = wsi_feature
        return output_dict

if __name__ == '__main__':
    import yaml
    with open("/home/auwqh/code/CLIP-MIL/examples/config_numhead4/clip_group_ensemble1.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    model = CLIP_MIL(
        **config["model"]
    )
    model.train()
    output = model(torch.randn(1, 1024, 512))
    for k, v in output.items():
        if v is not None:
            print(k, v.shape)
        else:
            print(k)
