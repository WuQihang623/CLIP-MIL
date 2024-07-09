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
            texts: dict,
            use_CoOP: bool,
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
        self.template = template

        if self.use_CoOP:
            prompts = []
            for class_name, descriptions in texts.items():
                if descriptions is None:
                    prompts.append(f"{template} {class_name}.")
                else:
                    prompts.append(f"{template} {class_name}, which shows {random.choice(descriptions)}")
            prompts = self.tokenizer(prompts).to(self.device)
            self.eos_postion = prompts.argmax(dim=-1)
            with torch.no_grad():
                embedding = self.token_embedding(prompts)

            n_ctx = len(template.split(" "))
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.ctx_embedding = nn.Parameter(embedding[0, 1: 1 + n_ctx, :])


    def forward_fixed(self):
        prompts = []
        for class_name, descriptions in self.texts.items():
            if descriptions is None:
                prompts.append(f"{self.template} {class_name}.")
            else:
                prompts.append(f"{self.template} {class_name}, which shows {random.choice(descriptions)}")
        prompts = self.tokenizer(prompts, context_length=self.context_length).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(prompts)
        eos_position = prompts.argmax(dim=-1)
        return embedding, eos_position

    def forward_CoOP(self):
        ctx = self.ctx_embedding
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_classes, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat([
            prefix,
            ctx,
            suffix
        ], dim=1)
        return prompts, self.eos_postion

    def forward(self):
        if self.use_CoOP:
            return self.forward_CoOP()
        else:
            return self.forward_fixed()



class GroupPooling(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            num_prototype: int,
            pooling_strategy: str,
            norm_layer: nn.Module=nn.LayerNorm,
            question: torch.Tensor=None,
            question_grad: bool=False
    ):
        super(GroupPooling, self).__init__()
        self.pooling_strategy = pooling_strategy

        if pooling_strategy == "first_token":
            weights = torch.zeros((1, num_prototype))
            weights[0, 0] = 1
            self.prototype_weight = nn.Parameter(weights)
            self.prototype_weight.requires_grad = False
        elif pooling_strategy == "mean":
            weights = torch.ones((1, num_prototype))
            self.prototype_weight = nn.Parameter(weights)
            self.prototype_weight.requires_grad = False
        elif pooling_strategy == "question":
            self.quetion_embedding = nn.Parameter(question, requires_grad=question_grad)
            self.question_attention = QuestionTransformerBlock(feat_dim=feat_dim, num_head=4, dropout=0.1)
        else:
            raise ValueError("Invalid pooling strategy")
        # self.pre_assign_attention = CrossAttnBlock(dim=feat_dim, num_heads=1)
        self.assign_attention = AssignAttention(dim=feat_dim, num_heads=1)
        self.image_norm = norm_layer(feat_dim)
        self.text_norm = norm_layer(feat_dim)
        # self.mlp = Mlp(in_features=feat_dim, hidden_features=feat_dim // 4, out_features=feat_dim)

    def forward(self, image_features, text_features):
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)

        image_features = self.image_norm(image_features)
        text_features = self.text_norm(text_features)

        new_image_features, attn = self.assign_attention(text_features, image_features)

        if self.pooling_strategy == "mean" or self.pooling_strategy == "first_token":
            prototype_weight = self.prototype_weight / torch.sum(self.prototype_weight)
            new_image_features = new_image_features.squeeze(0)
            new_image_features = prototype_weight @ new_image_features
        elif self.pooling_strategy == "question":
            new_image_features = self.question_attention(self.quetion_embedding, new_image_features)
            new_image_features = new_image_features[:, :1, :].squeeze(0)
        return new_image_features, attn


class SimilarityPooling(nn.Module):
    def __init__(
            self,
            num_prototype: int,
            pooling_strategy: str,
            feat_dim: int, question:
            torch.Tensor=None,
            question_grad: bool = False
    ):
        super(SimilarityPooling, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pooling_strategy = pooling_strategy

        if pooling_strategy == "first_token":
            weights = torch.zeros((1, num_prototype))
            weights[0, 0] = 1
            self.prototype_weight = nn.Parameter(weights)
            self.prototype_weight.requires_grad = False
        elif pooling_strategy == "attention_gate":
            self.attention = Attn_Net_Gated(feat_dim, feat_dim, dropout=True)
            self.rho = nn.Sequential(*[
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feat_dim, feat_dim)
            ])
        elif pooling_strategy == "question":
            self.quetion_embedding = nn.Parameter(question, requires_grad=question_grad)
            self.question_attention = QuestionTransformerBlock(feat_dim=feat_dim, num_head=4, dropout=0.1)

        else:
            raise ValueError("Unknown pooling strategy: {}".format(pooling_strategy))

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

        if self.pooling_strategy == "attention_gate":
            image_features = self.attention(image_features.squeeze(0))
            image_features = self.rho(image_features)
        elif self.pooling_strategy == "first_token":
            prototype_weight = self.prototype_weight / torch.sum(self.prototype_weight)
            image_features = prototype_weight @ image_features
        elif self.pooling_strategy == "question":
            image_features = self.question_attention(self.quetion_embedding, image_features)
        else:
            raise ValueError("Unknown pooling strategy: {}".format(self.pooling_strategy))
        return image_features, weights


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
            pooling_strategy: str,
            device: str,
            drop_instance: float=0.,
            question_grad=False,
            **kwargs
    ):
        super(CLIP_MIL, self).__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.feat_dim = feat_dim
        self.adaptor = Adapter(feat_dim, hidden_dim=feat_dim//4)
        self.text_encoder = clip.load_text_encoder(name=text_enc_name, device=self.device)

        self.drop_insatnce = nn.Dropout1d(drop_instance)

        self.pooling_method = pooling_method
        self.use_bag_prompt = use_bag_prompt
        self.use_cls_prompt = use_cls_prompt

        # set pooling method
        if pooling_method == "attention":
            assert self.use_cls_prompt is False and self.use_bag_prompt is False
            self.instance_pooling = Attn_Net_Gated(feat_dim, feat_dim, dropout=True)
        elif pooling_method == "mean":
            assert self.use_cls_prompt is False and self.use_bag_prompt is False
            self.instance_pooling = MeanPooling()
        elif pooling_method == "similarity" or pooling_method == "group":
            instance_template = kwargs["instance_template"]
            instance_texts = kwargs["instance_texts"]
            assert instance_template is not None and instance_texts is not None
            self.instance_token_embedding = PromptEmbedding(
                token_embedding=self.text_encoder.token_embedding,
                context_length=77,
                template=instance_template,
                texts=instance_texts,
                use_CoOP=use_CoOP,
                device=self.device,
            )
            if pooling_method == "similarity":
                if pooling_strategy == "question":
                    question = ["what is the degree of staining in the tumor area in this pathological image."]
                    question = clip.tokenize(question).to(self.device)
                    eos = question.argmax(dim=-1)
                    with torch.no_grad():
                        question = self.text_encoder.token_embedding(question)
                        question = self.text_encoder(question, eos).unsqueeze(0)
                else:
                    question = None
                self.instance_pooling = SimilarityPooling(
                    num_prototype=len(instance_texts), pooling_strategy=pooling_strategy, feat_dim=feat_dim, question=question, question_grad=question_grad
                )
            elif pooling_method == "group":
                if pooling_strategy == "question":
                    question = ["what is the degree of staining in the tumor area in this pathological image."]
                    question = clip.tokenize(question).to(self.device)
                    eos = question.argmax(dim=-1)
                    with torch.no_grad():
                        question = self.text_encoder.token_embedding(question)
                        question = self.text_encoder(question, eos).unsqueeze(0)
                else:
                    question = None
                self.instance_pooling = GroupPooling(
                    num_prototype=len(instance_texts), pooling_strategy=pooling_strategy, feat_dim=feat_dim, question=question, question_grad=question_grad
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
                device=self.device,
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
                device=self.device,
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
        if self.training:
            image_features = self.drop_insatnce(image_features)
            image_features = image_features[torch.sum(image_features, dim=-1)!=0].unsqueeze(0)

        output_dict = {}
        if self.pooling_method == "attention" or self.pooling_method == "mean":
            wsi_feature = self.instance_pooling(image_features)
        elif self.pooling_method == "similarity" or self.pooling_method == "group":
            instance_text_embedding, instance_eos_position = self.instance_token_embedding()
            instance_text_features = self.text_encoder(instance_text_embedding, instance_eos_position)
            wsi_feature, instance_attn = self.instance_pooling(image_features, instance_text_features)
            output_dict["instance_attention"] = instance_attn
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} is not implemented")

        wsi_feature = self.adaptor(wsi_feature)

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
    import yaml
    with open("/home/auwqh/code/CLIP-MIL/examples/config/description/clip_group_question.yaml", 'r') as f:
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

    # clip_model = clip.load_text_encoder(name="ViT-B/32", device="cpu")
    # prompt_embedding = PromptEmbedding(
    #         token_embedding=clip_model.token_embedding,
    #         context_length=77,
    #         template="a photo of a",
    #         texts={"cat": None, "dog": None},
    #         use_CoOP=True,
    #         device="cpu",
    #     )
    # prompts = prompt_embedding()

    # question = torch.randn((1, 1, 512))
    # model = GroupPooling(feat_dim=512, num_prototype=5, pooling_strategy="question", question=question)
    # text_embedding = torch.randn((1, 5, 512))
    # image_embedding = torch.randn((1, 1024, 512))
    # out, _ = model(image_embedding, text_embedding)
    # print(out.shape)
