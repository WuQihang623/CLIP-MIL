import torch
import random
import torch.nn as nn
import numpy as np

from src import clip


class PromptEmbedding(nn.Module):
    def __init__(
            self,
            token_embedding,
            context_length: int,
            template: str,
            texts: dict,
            use_CoOP: bool,
            trainable_prompt: bool,
            device: str,
    ):
        super(PromptEmbedding, self).__init__()
        """
        Initialize the LearnableTokenCLIPModel.
        Initialize the class_token by leveraging the specific prompt for each category, while retaining the original SOS (Start of Sequence) and CLS, as well as EOS (End of Sequence) tokens to preserve the integrity of sequence signals for guiding subsequent model processing.
        :param num_learnable_tokens:
            The number of learnable tokens. These tokens will be inserted into the input text embeddings and optimized during training.
        :param templates:
            A str of templates . These templates will be used to generate specific text inputs.
            Example: 'a histopathological image of '
        :param texts:
            A dictionary containing text descriptions for instance categories. The keys are category names, and the values are lists of multiple descriptions for each category.
            Example: {"tumor area": ["Flat, plate-like cells with a centrally located nucleus.", "Elongated cells with a basally located, oval-shaped nucleus."],
                      "stroma area": ["Calcified matrix with embedded osteocytes in lacunae, connected by canaliculi."]}
        """

        self.device = device

        self.texts = texts
        self.n_classes = len(texts)
        self.context_length = context_length
        self.tokenizer = clip.tokenize
        self.token_embedding = token_embedding

        self.use_CoOP = use_CoOP
        self.trainable = trainable_prompt

        if self.use_CoOP is False:
            assert self.trainable is False

        if self.use_CoOP:
            if trainable_prompt:
                prompts = []
                for class_name, descriptions in texts.items():
                    if descriptions is None:
                        prompts.append(f"{class_name}.")
                    else:
                        prompts.append(f"{class_name}, which shows {random.choice(descriptions)}")
                prompts = self.tokenizer(prompts).to(self.device)
                self.eos_postion = prompts.argmax(dim=-1)
                with torch.no_grad():
                    embedding = self.token_embedding(prompts)
                self.ctx_embedding = nn.Parameter(embedding)
            else:
                prompt = self.tokenizer([template for _ in range(self.n_classes)]).to(self.device)
                with torch.no_grad():
                    embedding = self.token_embedding(prompt)

                self.num_learnable_tokens = prompt.argmax(dim=-1)[0] - 1 # take out SOS
                ctx_embedding = embedding[:, 1: self.num_learnable_tokens + 1, :]
                self.ctx_embedding = nn.Parameter(ctx_embedding)  # to be optimized

    def forward_fixed(self):
        suffix_text = []
        for class_name, descriptions in self.texts.items():
            if descriptions is None:
                suffix_text.append(f"{class_name}.")
            else:
                suffix_text.append(f"{class_name}, which shows {random.choice(descriptions)}")
        prompt = self.tokenizer(suffix_text, context_length=self.context_length).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(prompt)
        eos_position = prompt.argmax(dim=-1)
        return embedding, eos_position


    def forward_trainable(self):
        ctx_embedding = self.ctx_embedding
        return ctx_embedding, self.eos_postion

    def forward_untrainable(self):
        suffix_text = []
        for class_name, descriptions in self.texts.items():
            if descriptions is None:
                suffix_text.append(f"{class_name}.")
            else:
                suffix_text.append(f"{class_name}, which shows {random.choice(descriptions)}")
        prompt = self.tokenizer(suffix_text, context_length=self.context_length - self.num_learnable_tokens).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(prompt)
        prefix_token = embedding[:, :1, :] # SOS
        suffix_token = embedding[:, 1:, :] # class token and EOS

        ctx = self.ctx_embedding

        token = torch.cat([prefix_token, ctx, suffix_token], dim=1)

        eos_position = self.num_learnable_tokens + prompt.argmax(dim=-1)
        return token, eos_position

    def forward(self):
        if self.use_CoOP:
            if self.trainable:
                return self.forward_trainable()
            else:
                return self.forward_untrainable()
        else:
            return self.forward_fixed()

class Adapter(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, ratio=0.2):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ratio = ratio

    def forward(self, x):
        return self.ratio * self.fc(x) + (1 - self.ratio) * x

class PromptGuidePooling(nn.Module):
    def __init__(self, num_prototype: int, learnable_pooling: bool):
        super(PromptGuidePooling, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        weights = torch.zeros((1, num_prototype))
        weights[0, 0] = 1
        self.prototype_weight = nn.Parameter(weights)
        if learnable_pooling is False:
            self.prototype_weight.requires_grad = False

    def forward(self, image_features, text_features):
        image_features = image_features.squeeze(0)
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features_norm @ text_features_norm.t()

        weights = logits_per_image.softmax(dim=-1)
        # image_features = weights.T @ image_features / torch.sum(weights, dim=0, keepdim=True).T
        # image_features = weights.T @ image_features / torch.tensor(image_features.shape[0], device=image_features.device)
        image_features = weights.T @ image_features / torch.sqrt(torch.tensor(image_features.shape[0], device=image_features.device))

        prototype_weight = self.prototype_weight / torch.sum(self.prototype_weight)
        image_features = prototype_weight @ image_features
        return image_features, weights


class PromptAttentionPooling(nn.Module):
    def __init__(self, feat_dim, num_heads, num_prototype: int, learnable_pooling: bool):
        super(PromptAttentionPooling, self).__init__()
        assert feat_dim % num_heads == 0

        self.attention = nn.MultiheadAttention(feat_dim, num_heads, dropout=0.1, batch_first=True)

        self.out_proj = nn.Linear(feat_dim, feat_dim)

        weights = torch.zeros((1, num_prototype))
        weights[0, 0] = 1
        self.prototype_weight = nn.Parameter(weights)
        if learnable_pooling is False:
            self.prototype_weight.requires_grad = False

    def forward(self, image_features, text_features):
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)

        attn_output, attn = self.attention(text_features, image_features, image_features)

        attn_output = self.out_proj(attn_output).squeeze(0)

        prototype_weight = self.prototype_weight / torch.sum(self.prototype_weight)
        attn_output = prototype_weight @ attn_output
        return attn_output, attn

class AttentionPooling(nn.Module):
    def __init__(self, feat_dim):
        super(AttentionPooling, self).__init__()
        self.L = feat_dim
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.BatchNorm1d(self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, image_features):
        image_features = image_features.squeeze(0)
        attn = self.attention(image_features)
        attn = torch.transpose(attn, 1, 0)
        attn = torch.softmax(attn, dim=1)

        image_features = torch.mm(attn, image_features)
        return image_features

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, image_features):
        return torch.mean(image_features.squeeze(0), dim=0, keepdim=True)

class CLIP_MIL(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            num_classes: int,
            text_enc_name: str,
            pooling_method: str,
            use_bag_prompt: bool,
            use_cls_prompt: bool,
            use_CoOP: bool,
            trainable_prompt: bool,
            learnable_pooling: bool,
            device: str,
            **kwargs
    ):
        super(CLIP_MIL, self).__init__()
        self.device = device
        self.feat_dim = feat_dim
        self.adaptor = Adapter(feat_dim, hidden_dim=feat_dim//4, ratio=0.2)
        self.text_encoder = clip.load_text_encoder(name=text_enc_name, device=device)

        self.pooling_method = pooling_method
        self.use_bag_prompt = use_bag_prompt
        self.use_cls_prompt = use_cls_prompt

        # set pooling method
        if pooling_method == "attention":
            assert self.use_cls_prompt is False and self.use_bag_prompt is False
            self.instance_pooling = AttentionPooling(feat_dim)
        elif pooling_method == "mean":
            assert self.use_cls_prompt is False and self.use_bag_prompt is False
            self.instance_pooling = MeanPooling()
        elif pooling_method == "promptguide" or pooling_method == "promptattention":
            instance_template = kwargs["instance_template"]
            instance_texts = kwargs["instance_texts"]
            assert instance_template is not None and instance_texts is not None
            self.instance_token_embedding = PromptEmbedding(
                token_embedding=self.text_encoder.token_embedding,
                context_length=77,
                template=instance_template,
                texts=instance_texts,
                use_CoOP=use_CoOP,
                trainable_prompt=trainable_prompt,
                device=device,
            )
            if pooling_method == "promptguide":
                self.instance_pooling = PromptGuidePooling(
                    num_prototype=len(instance_texts), learnable_pooling=learnable_pooling
                )
            else:
                self.instance_pooling = PromptAttentionPooling(
                    feat_dim=feat_dim, num_heads=8, num_prototype=len(instance_texts), learnable_pooling=learnable_pooling
                )
        else:
            raise NotImplementedError(f"Pooling method {pooling_method} is not implemented")

        if use_bag_prompt:
            bag_template = kwargs["bag_template"]
            bag_texts = kwargs["bag_texts"]
            assert bag_template is not None and bag_texts is not None
            self.bag_token_embedding = PromptEmbedding(
                token_embedding=self.text_encoder.token_embedding,
                context_length=77,
                template=bag_template,
                texts=bag_texts,
                use_CoOP=use_CoOP,
                trainable_prompt=trainable_prompt,
                device=device,
            )
            self.bag_prompt_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if use_cls_prompt:
            cls_template = kwargs["cls_template"]
            cls_texts = kwargs["cls_texts"]
            assert cls_template is not None and cls_texts is not None
            self.cls_token_embedding = PromptEmbedding(
                token_embedding=self.text_encoder.token_embedding,
                context_length=77,
                template=cls_template,
                texts=cls_texts,
                use_CoOP=use_CoOP,
                trainable_prompt=trainable_prompt,
                device=device,
            )
            self.cls_prompt_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        else:
            self.cls_head = nn.Linear(feat_dim, num_classes)

    def _apply_consine_similarity(self, image_features, text_features, scale):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits = scale * image_features @ text_features.t()
        return logits

    def forward(self, image_features):
        image_features = self.adaptor(image_features)

        output_dict = {}
        if self.pooling_method == "attention" or self.pooling_method == "mean":
            wsi_feature = self.instance_pooling(image_features)
        elif self.pooling_method == "promptguide" or self.pooling_method == "promptattention":
            instance_text_embedding, instance_eos_position = self.instance_token_embedding()
            instance_text_features = self.text_encoder(instance_text_embedding, instance_eos_position)
            wsi_feature, instance_attn = self.instance_pooling(image_features, instance_text_features)
            output_dict["instance_attention"] = instance_attn
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} is not implemented")

        if self.use_bag_prompt:
            bag_text_features, bag_eos_position = self.bag_token_embedding()
            bag_text_features = self.text_encoder(bag_text_features, bag_eos_position)
            scale = self.bag_prompt_logit_scale.exp()
            bag_logits = self._apply_consine_similarity(wsi_feature, bag_text_features, scale)
            output_dict["bag_logits"] = bag_logits
        if self.use_cls_prompt:
            cls_text_features, cls_eos_position = self.cls_token_embedding()
            cls_text_features = self.text_encoder(cls_text_features, cls_eos_position)
            scale = self.cls_prompt_logit_scale.exp()
            cls_logits = self._apply_consine_similarity(wsi_feature, cls_text_features, scale)
        else:
            cls_logits = self.cls_head(wsi_feature)
        output_dict["cls_logits"] = cls_logits
        return output_dict

if __name__ == '__main__':
    # import yaml
    # with open("/home/auwqh/code/CLIP-MIL/save_weights/clip_mil_vit_b32_meanPooling/config.yaml", 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    #     f.close()
    # model = CLIP_MIL(
    #     **config["model"]
    # )
    # logits, inst_attn = model(torch.randn(1, 1024, 512))
    # print(logits.shape, inst_attn.shape)

    model = PromptAttentionPooling(feat_dim=512, num_heads=8, num_prototype=5, learnable_pooling=False)
    x = torch.randn((1, 1024, 512))
    y = torch.randn((5, 512))
    z, _ = model(x, y)
