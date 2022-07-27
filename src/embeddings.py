import copy
import torch
from torch import nn

from gpt2_dual_stream import GPT2VisualFeatEncoder

# from volta: https://github.com/e-bug/volta/blob/71dd78a67b13543eb9edcf7e3a3946f5c2f566f0/volta/embeddings.py
class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(VisualBertEmbeddings, self).__init__()

        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = self.word_embeddings #gpt-2 style

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.embd_pdrop)

        # Segment and position embedding for image features
        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        self.token_type_embeddings_visual = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.special_initialize()

    def special_initialize(self):
        # This is a bit unorthodox. The better way might be to add an initializer to AllenNLP.
        # This function is used to init the token_type_embeddings_visual and position_embedding_visual, just in case.
        self.token_type_embeddings_visual.weight = torch.nn.Parameter(
            copy.deepcopy(self.token_type_embeddings.weight.data), requires_grad=True)
        self.position_embeddings_visual.weight = torch.nn.Parameter(
            copy.deepcopy(self.position_embeddings.weight.data), requires_grad=True)

    def forward(self, token_ids, image_feat, token_type_ids=None, position_ids=None):
        seq_length = token_ids.size(1)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if image_feat is not None:
            batch_size, num_boxes, _ = image_feat.shape
            visual_embeddings = self.projection(image_feat)
        
            visual_embeddings_type = torch.ones(
                image_feat.size()[:-1], dtype=torch.long, device=image_feat.device
            )
            token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)

            position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long, device=image_feat.device)
            position_embeddings_visual = self.position_embeddings_visual(position_ids_visual)

            v_embeddings = visual_embeddings + position_embeddings_visual + token_type_embeddings_visual

            # Concat the two:
            vl_embeddings = torch.cat((embeddings, v_embeddings), dim=1)  # concat visual embeddings after attentions
            vl_embeddings = self.layer_norm(vl_embeddings)
            vl_embeddings = self.dropout(vl_embeddings)
            embeddings, v_embeddings = vl_embeddings.split([embeddings.size(1), v_embeddings.size(1)], dim=1)
        else:
            v_embeddings = None
            embeddings = self.layer_norm(embeddings)
            embeddings = self.dropout(embeddings)

        return embeddings, v_embeddings

# from volta: https://github.com/e-bug/volta/blob/71dd78a67b13543eb9edcf7e3a3946f5c2f566f0/volta/embeddings.py
class UniterEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(UniterEmbeddings, self).__init__()

        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = self.word_embeddings #gpt-2 style

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.embd_pdrop)

        self.image_embeddings = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        self.image_token_type_embeddings = self.token_type_embeddings

        self.image_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.v_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.v_dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, token_ids, image_feat, token_type_ids=None, position_ids=None):
        seq_length = token_ids.size(1)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        if image_feat is not None:
            batch_size, num_boxes, _ = image_feat.shape
            img_embeddings = self.image_layer_norm(self.image_embeddings(image_feat))
            img_type_ids = torch.ones_like(image_feat[:, :, 0].long())
            v_token_type_embeddings = self.image_token_type_embeddings(img_type_ids)
            v_embeddings = img_embeddings + v_token_type_embeddings
            v_embeddings = self.v_layer_norm(v_embeddings)
            v_embeddings = self.v_dropout(v_embeddings)
        else:
            v_embeddings = None

        return embeddings, v_embeddings

class LXMERTEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(LXMERTEmbeddings, self).__init__()

        self.embed_dim = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.visn_fc = GPT2VisualFeatEncoder(config)

    def forward(self, token_ids, image_feat, token_type_ids=None, position_ids=None):
        seq_length = token_ids.size(1)

        inputs_embeds = self.word_embeddings(token_ids)
        position_embeds = self.wpe(position_ids)
        embeddings = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            embeddings = embeddings + token_type_embeds

        embeddings = self.drop(embeddings)

        if image_feat is not None:
            v_embeddings = self.visn_fc(image_feat)
        else:
            v_embeddings = None

        return embeddings, v_embeddings

EMBEDDING_TYPES = {"visualbert": VisualBertEmbeddings, "uniter": UniterEmbeddings, "lxmert": LXMERTEmbeddings}
