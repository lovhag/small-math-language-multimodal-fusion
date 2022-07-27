from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from train import AutoRegressiveTransformer, roll_columns
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from embeddings import EMBEDDING_TYPES, VisualBertEmbeddings, UniterEmbeddings

# LikeVisualBERT here refers to that GPT2 is adapted to be multimodal in a single-stream fashion like VisualBERT
class GPT2ConfigLikeVisualBERT(GPT2Config):
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
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        visual_embedding_dim=None,
        visual_embedding_type=None,
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
        self.visual_embedding_type = visual_embedding_type

class GPT2ModelLikeVisualBERT(GPT2Model):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super(GPT2Model, self).__init__(config)

        self.embed_dim = config.hidden_size

        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        
        # for visual features
        self.vl_embeddings = EMBEDDING_TYPES[config.visual_embedding_type](config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vl_embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.vl_embeddings.word_embeddings = new_embeddings

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
        use_cache=None,
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
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")

            # adapt attention mask to visual embeds as well
            if visual_embeds is not None:
                # add a visual attention mask
                if visual_attention_mask is None:
                    visual_attention_mask = torch.ones(visual_input_shape, device=device)

            if visual_attention_mask is not None:
                attention_mask = torch.cat((visual_attention_mask, attention_mask), dim=-1)

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

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        hidden_states, visual_embeddings = self.vl_embeddings(token_ids=input_ids, image_feat=visual_embeds, token_type_ids=token_type_ids, position_ids=position_ids)
        
        # both visualbert and uniter simply concat their embeddings
        if visual_embeds is not None:
            hidden_states = torch.cat((visual_embeddings, hidden_states), dim=1)
            assert visual_input_shape[0] == input_shape[0] and len(visual_input_shape) == len(input_shape), "Input and visual shapes do not match"
            output_shape = (input_shape[0], input_shape[1]+visual_input_shape[1]) + (hidden_states.size(-1),)
        else:
            output_shape = input_shape + (hidden_states.size(-1),)

        # nothing should be different from the original GPT2 code after this point
        # original code: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/gpt2/modeling_gpt2.py#L668

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GPT2LMHeadModelLikeVisualBERT(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelLikeVisualBERT(config)
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

            # visual input (otherwise appended first) should also be removed if past
            visual_embeds = None

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

class AutoRegressiveTransformerLikeVisualBERT(AutoRegressiveTransformer):
    def __init__(self, hparams, gpt_model=None):
        super().__init__(hparams)

        self.visual_embedding_dim = hparams["visual_embedding_dim"]

        if gpt_model is not None:
            self.model = gpt_model
        else:
            visual_config = GPT2ConfigLikeVisualBERT(
                            vocab_size=hparams["vocab_size"],
                            n_positions=hparams["n_positions"],
                            n_embd=hparams["n_embd"],
                            n_layer=hparams["n_layer"],
                            n_inner=hparams["n_inner"],
                            n_head=hparams["n_head"],
                            visual_embedding_dim=hparams["visual_embedding_dim"],
                            visual_embedding_type=hparams["visual_embedding_type"]
                            )
            self.model = GPT2LMHeadModelLikeVisualBERT(visual_config)

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
