import torch
import random
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from transformers import AutoTokenizer, CLIPConfig

class LearnableTokenCLIPModel(nn.Module):
    def __init__(self, model_name, num_learnable_tokens, instance_templates: list, instance_texts: dict, bag_templates: list, bag_texts: dict, device: str):
        """
        Initialize the LearnableTokenCLIPModel.
        :param model_name:
            The name or path of the pretrained CLIP model. This will be used to load the model configuration, tokenizer, and text encoder.
        :param num_learnable_tokens:
            The number of learnable tokens. These tokens will be inserted into the input text embeddings and optimized during training.
        :param instance_templates:
            A list of templates for instance categories. These templates will be used to generate specific text inputs.
            Example: ['a histopathological image of {}', 'a microscopic image of {} in tissue,
                      'a high magnification image of {}', 'an immunohistochemical staining of {}']
        :param instance_texts:
            A dictionary containing text descriptions for instance categories. The keys are category names, and the values are lists of multiple descriptions for each category.
            Example: {"tumor area": ["Flat, plate-like cells with a centrally located nucleus.", "Elongated cells with a basally located, oval-shaped nucleus."],
                      "stroma area": ["Calcified matrix with embedded osteocytes in lacunae, connected by canaliculi."]}
        :param bag_templates:
            A list of templates for bag categories. These templates will be used to generate specific text inputs.
            Example: ["An immunohistochemistry WSI with a TPS score of {}"]
        :param bag_texts:
            A dictionary containing text descriptions for bag categories. The keys are category names, and the values are lists of multiple descriptions for each category.
            Example: {"less than 1%": ["negligible detection of the target biomarker in the tumor area, indicating extremely low levels of immune cell infiltration"],
                      "between 1% to 50%": ["mild to moderate expression of the target biomarker in some tumor cells, suggesting limited immune response activity",
                      "more than 50%": ["over half of the tumor cells exhibiting strong expression of the target biomarker"]]}
        """
        super(LearnableTokenCLIPModel, self).__init__()
        self.config = CLIPConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.device = device
        self.bag_learnable_tokens = nn.Parameter(
            torch.rand(num_learnable_tokens, self.config.text_config.hidden_size)
        ).to(self.device)
        self.instance_learnable_tokens = nn.Parameter(
            torch.rand(num_learnable_tokens, self.config.text_config.hidden_size)
        ).to(self.device)

        self.instance_templates = instance_templates
        self.instance_texts = instance_texts

        self.bag_templates = bag_templates
        self.bag_texts = bag_texts
        self.device = device

    def get_instance_text_embedding(self):
        texts = []
        for propotype, descriptions in self.instance_texts.items():
            text = random.choice(self.instance_templates).format(propotype) + random.choice([", showing{}.", ", which shows {}."]).format(random.choice(descriptions))
            texts.append(text)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        learnable_tokens = self.instance_learnable_tokens.unsqueeze(0).expand(inputs_ids.size(0), -1, -1)
        embeddings = self.text_encoder.get_input_embeddings()(input_ids=inputs_ids, position_ids=None)
        embeddings = torch.cat([embeddings, learnable_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), self.instance_learnable_tokens.size(0)), dtype=attention_mask.dtype).to(self.device)],
            dim=1
        )
        outputs = self.text_encoder(input_embeds=embeddings, attention_mask=attention_mask)
        return outputs.pooler_output

    def get_bag_text_embedding(self):
        texts = []
        for propotype, descriptions in self.bag_texts.items():
            text = random.choice(self.bag_templates).format(propotype) + random.choice([", showing{}.", ", which shows {}."]).format(random.choice(descriptions))
            texts.append(text)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        learnable_tokens = self.bag_learnable_tokens.unsqueeze(0).expand(inputs_ids.size(0), -1, -1)
        embeddings = self.text_encoder.get_input_embeddings()(input_ids=inputs_ids, position_ids=None)
        embeddings = torch.cat([embeddings, learnable_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((attention_mask.size(0), self.bag_learnable_tokens.size(0)), dtype=attention_mask.dtype).to(self.device)],
            dim=1
        )
        outputs = self.text_encoder(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.pooler_output

    def forward(self):
        instance_embedding = self.get_instance_text_embedding()
        bag_embdding = self.get_bag_text_embedding()
        return instance_embedding, bag_embdding


class Adaptor(nn.Module):
    def __init__(self, feat_dim):
        super(Adaptor, self).__init__()
        self.fc = nn.Linear(feat_dim, feat_dim)

    def forward(self, x):
        return x + self.fc(x)

class PromptGuidePooling(nn.Module):
    def __init__(self):
        super(PromptGuidePooling, self).__init__()
        pass

class CLIP_MIL(nn.Module):
    def __init__(self, feat_dim, text_enc_name, instance_text: list, bag_text: list):
        super(CLIP_MIL, self).__init__()
        self.feat_dim = feat_dim
        self.text_encoder = CLIPTextModel.from_pretrained(text_enc_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_enc_name)
        self.adaptor = Adaptor(feat_dim)
        self.instance_text = instance_text
        self.bag_text = bag_text

    def bag_level_prompt(self):
        pass

    def instance_level_prompt(self):
        pass

    def forward(self, x):
        x = self.adaptor(x)


if __name__ == '__main__':
    model = LearnableTokenCLIPModel(
        model_name="/home/auwqh/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/",
        num_learnable_tokens=10, instance_templates=['a histopathological image of {}', 'a microscopic image of {} in tissue',
                                                     'a high magnification image of {}', 'an immunohistochemical staining of {}'],
        instance_texts={"tumor area": ["Flat, plate-like cells with a centrally located nucleus.", "Elongated cells with a basally located, oval-shaped nucleus."],
                        "stroma area": ["Calcified matrix with embedded osteocytes in lacunae, connected by canaliculi."]},
        bag_templates=["An immunohistochemistry WSI with a TPS score of {}"],
        bag_texts={"less than 1%": ["negligible detection of the target biomarker in the tumor area, indicating extremely low levels of immune cell infiltration"],
                   "between 1% to 50%": ["mild to moderate expression of the target biomarker in some tumor cells, suggesting limited immune response activity"],
                   "more than 50%": ["over half of the tumor cells exhibiting strong expression of the target biomarker"]},
        device="cpu"
    )
    instance_embedding, bag_embdding = model()
    print(instance_embedding.shape)
    print(bag_embdding.shape)
