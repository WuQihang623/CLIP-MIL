import torch
import random
import torch.nn as nn
import numpy as np

from src import clip

class PromptLearner(nn.Module):
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
        super(PromptLearner, self).__init__()
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

class Adaptor(nn.Module):
    def __init__(self, feat_dim):
        super(Adaptor, self).__init__()
        self.fc = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.fc(x))

class PromptGuidePooling(nn.Module):
    def __init__(self, num_prototype: int, learnabel_pooling: bool):
        super(PromptGuidePooling, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        weights = torch.zeros((1, num_prototype))
        weights[0, 0] = 1
        self.prototype_weight = nn.Parameter(weights)
        if learnabel_pooling is False:
            self.prototype_weight.requires_grad = False

    def forward(self, image_features, text_features):
        image_features = image_features.squeeze(0)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features_norm @ text_features_norm.t()

        weights = logits_per_image.softmax(dim=0)
        image_features = weights.T @ image_features

        image_features = self.prototype_weight @ image_features
        return image_features, weights


class CLIP_MIL(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            text_enc_name: str,
            instance_template: str,
            instance_texts: dict,
            bag_template: str,
            bag_texts: dict,
            cls_template: str,
            cls_texts: dict,
            use_CoOP: bool,
            trainable_prompt: bool,
            learnabel_pooling: bool,
            device: str
    ):
        super(CLIP_MIL, self).__init__()
        self.feat_dim = feat_dim
        self.adaptor = Adaptor(feat_dim)

        self.text_encoder = clip.load_text_encoder(name=text_enc_name, device=device)
        self.device = device

        self.instance_texts = instance_texts
        self.bag_texts = bag_texts

        self.instance_promptor = PromptLearner(
            token_embedding=self.text_encoder.token_embedding,
            context_length=77,
            template=instance_template,
            texts=instance_texts,
            use_CoOP=use_CoOP,
            trainable_prompt=trainable_prompt,
            device=device,
        )

        self.cls_promptor = PromptLearner(
            token_embedding=self.text_encoder.token_embedding,
            context_length=77,
            template=cls_template,
            texts=cls_texts,
            use_CoOP=use_CoOP,
            trainable_prompt=trainable_prompt,
            device=device,
        )

        self.prompt_pooling = PromptGuidePooling(num_prototype=len(self.instance_texts), learnabel_pooling=learnabel_pooling)
        self.cls_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.bag_promptor = None
        self.bag_prompt_logit_scale = None
        if bag_template is not None:
            self.bag_promptor = PromptLearner(
                token_embedding=self.text_encoder.token_embedding,
                context_length=77,
                template=bag_template,
                texts=bag_texts,
                use_CoOP=use_CoOP,
                trainable_prompt=trainable_prompt,
                device=device,
            )
            self.bag_prompt_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features):
        # instance text prompt
        inst_text_token, inst_eos_pos = self.instance_promptor()
        inst_text_features = self.text_encoder(inst_text_token, inst_eos_pos)

        # class text prompt
        cls_text_token, cls_eos_pos = self.cls_promptor()
        cls_text_features = self.text_encoder(cls_text_token, cls_eos_pos)


        image_features = self.adaptor(image_features)
        image_features, inst_attn = self.prompt_pooling(image_features, inst_text_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        cls_text_features = cls_text_features / cls_text_features.norm(dim=1, keepdim=True)
        cls_logit_scal = self.cls_logit_scale.exp()
        cls_logits = cls_logit_scal * image_features @ cls_text_features.t()

        # bag text promt
        if self.bag_promptor is not None:
            bag_text_token, bag_eos_pos = self.bag_promptor()
            bag_text_features = self.text_encoder(bag_text_token, bag_eos_pos)
            bag_text_features = bag_text_features / bag_text_features.norm(dim=1, keepdim=True)
            bag_prompt_logit_scale = self.bag_prompt_logit_scale.exp()
            bag_prompt_logits = bag_prompt_logit_scale * image_features @ bag_text_features.t()
            return {"cls_logits": cls_logits, "bag_prompt_logits": bag_prompt_logits, "inst_attn": inst_attn}

        else:
            return {"cls_logits": cls_logits, "inst_attn": inst_attn}

if __name__ == '__main__':
    import yaml
    with open("/home/auwqh/code/CLIP-MIL/examples/config/clip_mil_vit_b32.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    model = CLIP_MIL(
        **config["model"]
    )
    logits, inst_attn = model(torch.randn(1, 1024, 512))
    print(logits.shape, inst_attn.shape)
