from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from train import AutoRegressiveTransformer, roll_columns
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# LikeLXMERT here refers to that GPT2 is adapted to be multimodal in a dual-stream fashion like LXMERT
class GPT2ConfigLikeLXMERT(GPT2Config):
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        visual_embedding_dim=None,
        l_layers=4, #sums to 12?
        x_layers=4,
        r_layers=4,
        **kwargs,
    ):
        super().__init__(vocab_size, 
                         n_positions, 
                         n_embd, 
                         n_layer, 
                         n_head,
                         n_inner,
                         activation_function,
                         resid_pdrop,
                         embd_pdrop,
                         attn_pdrop,
                         layer_norm_epsilon,
                         initializer_range,
                         summary_type,
                         summary_use_proj,
                         summary_activation,
                         summary_proj_to_labels,
                         summary_first_dropout,
                         scale_attn_weights,
                         use_cache,
                         bos_token_id,
                         eos_token_id,
                         scale_attn_by_inverse_layer_idx,
                         reorder_and_upcast_attn)
        
        self.visual_embedding_dim = visual_embedding_dim

        self.l_layers = l_layers
        self.r_layers = r_layers
        self.x_layers = x_layers

class GPT2LXRTXLayer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size

        # The cross-attention Layer
        self.cross_attention = GPT2Attention(config, is_cross_attention=True)
        self.ln_cross_attn = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.lang_ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lang_ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.visn_ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.visn_ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Self-attention Layers
        self.lang_self_att = GPT2Attention(config, layer_idx=layer_idx)
        self.visn_self_att = GPT2Attention(config, layer_idx=layer_idx)

        # Intermediate and Output Layers (FFNs)
        self.lang_mlp = GPT2MLP(inner_dim, config)
        self.visn_mlp = GPT2MLP(inner_dim, config)

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask,
                      lang_layer_past, visn_layer_past):
        # self attention
        lang_residual = lang_feats
        lang_hidden_states = self.lang_ln_1(lang_feats)
        lang_attn_outputs = self.lang_self_att(lang_hidden_states, attention_mask=lang_attention_mask, layer_past=lang_layer_past, use_cache=True)
        lang_attn_output = lang_attn_outputs[0]
        lang_outputs = lang_attn_outputs[1:]
        lang_hidden_states = lang_attn_output + lang_residual # residual connection

        if visn_feats is not None:
            visn_residual = visn_feats
            visn_hidden_states = self.visn_ln_1(visn_feats)
            visn_attn_outputs = self.visn_self_att(visn_hidden_states, attention_mask=visn_attention_mask, layer_past=visn_layer_past, use_cache=True)
            visn_attn_output = visn_attn_outputs[0]
            visn_outputs = visn_attn_outputs[1:]
            visn_hidden_states = visn_attn_output + visn_residual

        # cross attention
        if visn_feats is not None: # TODO: investigate whether the handling of no visual features should be done better in cross-modal! 
            # test: no cross modal layers at all and see
            # test: still add visual features as encoder state for text part, when that is of interest
            # should be able to handle it though (encoder hidden state None)!
            lang_residual = lang_hidden_states
            lang_cross_attn_outputs = self.cross_attention(lang_hidden_states, attention_mask=lang_attention_mask, encoder_hidden_states=visn_hidden_states, encoder_attention_mask=visn_attention_mask)
            lang_attn_output = lang_cross_attn_outputs[0]
            lang_hidden_states = lang_residual + lang_attn_output
            lang_outputs = lang_outputs + lang_cross_attn_outputs[2:]

            # cannot run this part of the model if we have a past (attention mask for language and input for language won't mask due to some in past)
            if lang_layer_past is None:
                visn_residual = visn_hidden_states
                visn_cross_attn_outputs = self.cross_attention(visn_hidden_states, attention_mask=visn_attention_mask, encoder_hidden_states=lang_hidden_states, encoder_attention_mask=lang_attention_mask)
                visn_attn_output = visn_cross_attn_outputs[0]
                visn_hidden_states = visn_residual + visn_attn_output
                visn_outputs = visn_outputs + visn_cross_attn_outputs[2:]
            
        # last output stage
        lang_residual = lang_hidden_states
        lang_hidden_states = self.lang_ln_2(lang_hidden_states)
        lang_feed_forward_hidden_states = self.lang_mlp(lang_hidden_states)
        lang_hidden_states = lang_residual + lang_feed_forward_hidden_states
        lang_outputs = (lang_hidden_states,) + lang_outputs

        if visn_feats is not None:
            visn_residual = visn_hidden_states
            visn_hidden_states = self.visn_ln_2(visn_hidden_states)
            visn_feed_forward_hidden_states = self.visn_mlp(visn_hidden_states)
            visn_hidden_states = visn_residual + visn_feed_forward_hidden_states
            visn_outputs = (visn_hidden_states,) + visn_outputs
        else:
            visn_outputs = None

        return {"lang_outputs": lang_outputs, "visn_outputs": visn_outputs}
        
class GPT2VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Object feature encoding
        self.visn_fc = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        self.visn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.embd_pdrop)

    def forward(self, visn_input):
        x = self.visn_fc(visn_input)
        x = self.visn_layer_norm(x)

        output = self.dropout(x)
        return output


class GPT2LXRTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = GPT2VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        self.layer = nn.ModuleList(
            [GPT2Block(config, layer_idx=i) for i in range(self.num_l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [GPT2LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [GPT2Block(config, layer_idx=i) for i in range(self.num_r_layers)]
        )

    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None,
                past_key_values=None):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        if visn_feats is not None:
            visn_feats = self.visn_fc(visn_feats)

        presents = { "lang": (), "visn": (), "cross_lang": (), "cross_visn": ()} 

        # Run language layers
        for i, (block, layer_past) in enumerate(zip(self.layer, past_key_values["lang"])):
            outputs = block(lang_feats, attention_mask=lang_attention_mask, layer_past=layer_past, use_cache=True)
            lang_feats = outputs[0]
            presents["lang"] = presents["lang"] + (outputs[1],)

        # Run relational layers
        if visn_feats is not None:
            for i, (block, layer_past) in enumerate(zip(self.r_layers, past_key_values["visn"])):
                outputs = block(visn_feats, attention_mask=visn_attention_mask, layer_past=layer_past, use_cache=True)
                visn_feats = outputs[0]
                presents["visn"] = presents["visn"] + (outputs[1],)

        # Run cross-modality layers
        for i, (layer_module, lang_layer_past, visn_layer_past) in enumerate(zip(self.x_layers, past_key_values["cross_lang"], past_key_values["cross_visn"])):
            outputs = layer_module(lang_feats, lang_attention_mask,
                                   visn_feats, visn_attention_mask,
                                   lang_layer_past, visn_layer_past) # will always use cache
            lang_feats = outputs["lang_outputs"][0]
            presents["cross_lang"] = presents["cross_lang"] + (outputs["lang_outputs"][1],)

            if visn_feats is not None:
                visn_feats = outputs["visn_outputs"][0]
                presents["cross_visn"] = presents["cross_visn"] + (outputs["visn_outputs"][1],)

        return lang_feats, visn_feats, presents


class GPT2ModelLikeLXMERT(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # vision specific
        self.encoder = GPT2LXRTEncoder(config)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=True, # must save past values to be able to generate correctly?
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if visual_embeds is not None:
            visual_input_shape = visual_embeds.size()[:-1]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = {"lang": tuple([None] * self.encoder.num_l_layers), 
                               "visn": tuple([None] * self.encoder.num_r_layers),
                               "cross_lang": tuple([None] * self.encoder.num_x_layers),
                               "cross_visn": tuple([None] * self.encoder.num_x_layers)
                              }
        else:
            past_length = past_key_values["lang"][0][0].size(-2)
            
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")

            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Process the visual attention mask as well
        if visual_embeds is not None:
            if visual_attention_mask is None:
                visual_attention_mask = torch.ones(visual_input_shape, device=device)

            visual_attention_mask = visual_attention_mask.view(batch_size, -1)
            visual_attention_mask = visual_attention_mask[:, None, None, :]

            visual_attention_mask = visual_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            visual_attention_mask = (1.0 - visual_attention_mask) * -10000.0

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        
        if visual_embeds is not None and past_length == 0:
            assert visual_input_shape[0] == input_shape[0] and len(visual_input_shape) == len(input_shape), "Input and visual shapes do not match"
            output_shape = (input_shape[0], input_shape[1]+visual_input_shape[1]) + (hidden_states.size(-1),)
        else:
            output_shape = input_shape + (hidden_states.size(-1),)

        lang_feats, visn_feats, presents = self.encoder(lang_feats=hidden_states, 
                                              lang_attention_mask=attention_mask,
                                              visn_feats=visual_embeds,
                                              visn_attention_mask=visual_attention_mask,
                                              past_key_values=past_key_values)

        # do not output hidden states for visual if only interested in generating text (past_key_values exist)
        if visual_embeds is not None and past_length == 0:
            hidden_states = torch.cat((visn_feats, lang_feats), dim=1)
        else:
            hidden_states = lang_feats

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

class GPT2LMHeadModelLikeLXMERT(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelLikeLXMERT(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # overrides the parent function
        token_type_ids = kwargs.get("token_type_ids", None)
        
        visual_embeds = kwargs.get("visual_embeds", None)
        visual_attention_mask = kwargs.get("visual_attention_mask", None)

        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

            # always give visual embedding input, so no need to give past as well
            past["visn"] = tuple([None] * self.transformer.encoder.num_r_layers)
            past["cross_visn"] = tuple([None] * self.transformer.encoder.num_x_layers)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "visual_embeds": visual_embeds,
            "visual_attention_mask": visual_attention_mask
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        lm_logits_for_loss = lm_logits.clone()

        # do not calculate the loss based on the visual embeddings
        if visual_embeds is not None:
            lm_logits_for_loss = lm_logits[:,visual_embeds.shape[1]:,:]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits_for_loss[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class AutoRegressiveTransformerLikeLXMERT(AutoRegressiveTransformer):
    def __init__(self, hparams, gpt_model=None):
        super().__init__(hparams)

        self.visual_embedding_dim = hparams["visual_embedding_dim"]

        if gpt_model is not None:
            self.model = gpt_model
        else:
            visual_config = GPT2ConfigLikeLXMERT(
                            vocab_size=hparams["vocab_size"],
                            n_positions=hparams["n_positions"],
                            n_embd=hparams["n_embd"],
                            n_layer=hparams["n_layer"],
                            n_inner=hparams["n_inner"],
                            n_head=hparams["n_head"],
                            visual_embedding_dim=hparams["visual_embedding_dim"],
                            l_layers=hparams["l_layers"],
                            r_layers=hparams["r_layers"],
                            x_layers=hparams["x_layers"]
                            )
            self.model = GPT2LMHeadModelLikeLXMERT(visual_config)

        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        generate_after_token_idxs = (batch["input_ids"] == self.generate_after_token).long().argmax(dim=1)

        # Align on generate_after_token for batch generation
        input_ids = F.pad(
            batch["input_ids"], 
            (0, generate_after_token_idxs.max() - generate_after_token_idxs.min(), 0, 0), 
            value=self.pad_token_idx
        )
        input_ids = roll_columns(input_ids, generate_after_token_idxs.max() - generate_after_token_idxs)
        attention_mask =  (input_ids != self.pad_token_idx).int().to(input_ids.device)

        if self.mask_data_type_specific_tokens:
            assert torch.all(batch["data_types"][0] == batch["data_types"]), "During validation all examples in batch must be of same data_type"
            bad_words_ids = [ [token_id] for token_id in self.bad_words_ids_per_data_type[batch["data_types"][0]] ]
        else:
            bad_words_ids = None
        gen_out = self.model.generate(
            input_ids[:, :generate_after_token_idxs.max()+1], 
            attention_mask=attention_mask[:, :generate_after_token_idxs.max()+1],
            visual_embeds=batch["visual_embeds"],
            visual_attention_mask=batch["visual_attention_mask"],
            max_length=input_ids.shape[1], 
            pad_token_id=self.pad_token_idx,
            bad_words_ids=bad_words_ids
        )

        is_prediction_correct = ((gen_out == input_ids) | (input_ids == self.pad_token_idx)).all(dim=1)
        metric = self.train_comp_accuracy if dataloader_idx == 0 else self.val_comp_accuracy
        metric(is_prediction_correct, torch.ones_like(is_prediction_correct, dtype=torch.bool))
