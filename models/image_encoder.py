class VisualPrompt(nn.Module):
    def __init__(self, prompt_size, image_size):
        super().__init__()
        self.prompt_size = prompt_size
        self.image_size = image_size
        self.prompt = nn.Parameter(torch.randn(3, prompt_size, prompt_size))  # Learnable patch

    def forward(self, images):
        batch_size, channels, height, width = images.size()  # Get dimensions of the image tensor

        # Ensure that the prompt can fit within the image dimensions
        if self.prompt_size > height or self.prompt_size > width:
            raise ValueError(f"Prompt size {self.prompt_size} is too large for the image size {height}x{width}")

        for i in range(batch_size):
            # Randomly select position to insert prompt ensuring it fits within the image
            x_pos = torch.randint(0, height - self.prompt_size, (1,)).item()
            y_pos = torch.randint(0, width - self.prompt_size, (1,)).item()

            # Insert the prompt at the chosen position
            images[i, :, x_pos:x_pos + self.prompt_size, y_pos:y_pos + self.prompt_size] = self.prompt

        return images

def create_gaussian_target_map(patch_position, hidden_size, seq_length):
    """
    Creates a Gaussian target map centered around the patch position in token space.
    """
    gaussian_map = torch.zeros(( seq_length,hidden_size))

    x_center, y_center = patch_position
    sigma = hidden_size / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))

    for i in range(seq_length):
        for j in range(hidden_size):
            gaussian_map[i, j] = torch.exp(-((i - x_center) ** 2 + (j - y_center) ** 2) / (2 * sigma ** 2))

    return gaussian_map

def compute_kl_loss(attention_weights, target_map):
    """
    Compute KL-Divergence between attention weights and the Gaussian target map.
    """
    attention_probs = nn.functional.softmax(attention_weights, dim=-1)
    target_probs = nn.functional.softmax(target_map, dim=-1)
    loss = nn.functional.kl_div(attention_probs.log(), target_probs, reduction='batchmean')
    return loss


class UnimoModel(nn.Module):
    def __init__(self, vision_config, text_config, add_pooling_layer=True,prompt_size=32):
        #print("Initializng UnimoModel of modelling_unimo.py")
        super(UnimoModel, self).__init__()
        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        self.visual_prompt = VisualPrompt(prompt_size, vision_config.image_size)  # 
        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # all
        self.encoder = UnimoEncoder(vision_config, text_config)

        self.device = vision_config.device
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        
        pixel_values=None,
        aux_values=None, 
        rcnn_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        prompt_loss=True
    ):
        #print("Inside forward of UnimoModel of modelling_unimo")
        # pre vision
        pixel_values = self.visual_prompt(pixel_values)
        # rcnn_values = self.visual_prompt(rcnn_values)
        # aux_values = self.visual_prompt(aux_values)
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            if hasattr(self.text_embeddings, "token_type_ids"):
                buffered_token_type_ids = self.text_embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)    # [None]*12

        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # all encoder
        encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        # print("Encoder output = ",encoder_outputs)
        sequence_output = encoder_outputs[0]
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

        kl_loss = None
        if prompt_loss:
        #     # Extract attention weights from vision encoder
            v_hidden_states = encoder_outputs.hidden_states[-1].detach().clone()
            _, h , w = v_hidden_states.shape
        #     # Get patch position and create target Gaussian map
            patch_position = (self.visual_prompt.prompt_size // 2, self.visual_prompt.prompt_size // 2)
            target_map = create_gaussian_target_map(patch_position, self.vision_config.hidden_size, seq_length)
            v_hidden_states = v_hidden_states.mean(dim=1)
            target_map = target_map.to(v_hidden_states.device)
            # print("Hidden state shape = ",v_hidden_states.shape )
            # target_map = target_map.unsqueeze(1).repeat(1, h, 1)
            # print("Target shape = ",target_map.shape)
        #     # Compute KL loss
            kl_loss = compute_kl_loss(v_hidden_states, target_map)

        if not return_dict:
            print("inside not reutrn dict")
            return (kl_loss,sequence_output, pooled_output) + encoder_outputs[1:]
        
        # if not return_dict:
        #     #print("Exiting forward of UnimoModel of modelling_unimo")
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]
        #print("Exiting forward of UnimoModel of modelling_unimo")
        
        return (kl_loss,BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ))
     
    def _init_text_weights(self, module):
        """Initialize the weights"""
        #print("Executing _init_text_weights of UnimoModel of modelling_unimo")
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        #print("Executing get_input_embeddings of UnimoModel of modelling_unimo")
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        #print("Executing set_input_embeddings of UnimoModel of modelling_unimo")
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        #print("Executing resize_token_embeddings of UnimoModel of modelling_unimo")
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        #print("Inside _get_resized_embeddings of UnimoModel of modelling_unimo")
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        #print("Exiting _get_resized_embeddings of UnimoModel of modelling_unimo")
        return new_embeddings
    
class UnimoEncoder(nn.Module):
    def __init__(self, vision_config, text_config):
        #print("Initializing UnimoEncoder of modelling_unimo")
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config

        self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])
    
    def forward(
        self,
        vision_embeds=None,
        text_embeds=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #print("Inside forward of UnimoEncoder of modelling_unimo")
        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        
        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )
            
            # vision
            # TODO: 9-12 layers past text as pkv to vision
            past_key_values = text_layer_output[-1] if idx >= 8 else None
            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                    vision_hidden_states,
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
            )
            vision_hidden_states = vision_layer_output[0]

            # text
            # TODO: 9-12 layers past vison qks to text
            last_hidden_state = vision_hidden_states if idx >= 8 else None
            output_qks = True if idx >= 7 else None
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.text_layer[idx]
            text_layer_output = text_layer_module(
                    text_hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    visual_hidden_state=last_hidden_state,
                    output_attentions=output_attentions,
                    output_qks=output_qks,
            )
            text_hidden_states = text_layer_output[0]
            # print("Output attentions = ",output_attentions)
            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1], )
                all_text_attentions = all_text_attentions + (text_layer_output[1], )
                if idx >= 8:
                    all_cross_attentions = all_cross_attentions + (text_layer_output[2],)  # Cross-attention weights
        
        if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )
        
        if not return_dict:
            #print("Exiting forward of UnimoEncoder of modelling_unimo")
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)
        #print("Exiting forward of UnimoEncoder of modelling_unimo")
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=text_hidden_states,
            hidden_states=all_vision_hidden_states,
            attentions=all_text_attentions,
            cross_attentions=all_cross_attentions
        )
    

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        #print("Initialzing CLIPEncoderLayer modelling_unimo.py")
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        #print("Inside forward of CLIPEncoderLayer of modelling_unimo.py")
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        #print("exiting forward of CLIPEncoderLayer of modelling_unimo.py")
        return outputs

class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        #print("Initializing CLIPAttention of modelling_unimo.py")
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        #print("Inside forward of CLIPAttention of modelling_unimo.py")
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )       
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        #print("Exiting forward of CLIPAttention of modelling_unimo.py")
        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, config):
        #print("Initializing CLIPMPL in modelling_unimo")
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        #print("Executing forward of CLIPMLP class of modelling_unimo.py")
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        #print("Initializing CLIPVisionEmbeddings in modelling_unimo.py")
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        # lilei:
        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        #print("Inside forward of CLIPVisionEmbeddings in modelling_unimo.py")
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # shape = [*, grid*grid, width]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # lilei
        embeddings = patch_embeds
        # embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # embeddings = embeddings + self.position_embedding(self.position_ids)

        # lilei:
        if aux_embeddings is not None:
            aux_embeds = []
            for aux_embedding in aux_embeddings:
                aux_embed = self.patch_embedding(aux_embedding)
                aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)    # 3*16, 768 3个子图
                aux_embeds.append(aux_embed)
            aux_embeds = torch.stack(aux_embeds) # bsz, 48, 768
            # aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)

        if rcnn_embeddings is not None:
            rcnn_embeds = []
            for rcnn_embedding in rcnn_embeddings:
                rcnn_embed = self.patch_embedding(rcnn_embedding)
                rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)    # 3*4, 768 3个子图
                rcnn_embeds.append(rcnn_embed)
            rcnn_embeds = torch.stack(rcnn_embeds) # bsz, 12, 768
            # rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
            embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        #print("Exiting forward of CLIPVisionEmbeddings in modelling_unimo.py")
        return embeddings

       
class UnimoForMaskedLM(nn.Module):
    def __init__(self, vision_config, text_config):
        #print("Initializing UnimoForMaskedLM in modelling_unimo")
        super().__init__()
        self.unimo = UnimoModel(vision_config, text_config)
        self.cls = UnimoOnlyMLMHead(text_config)
        self.config = text_config

        self.tie_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        
        pixel_values=None,
        aux_values=None, 
        rcnn_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        #print("Inside forward of UnimoForMaskedLM of modelling_unimo")
        kl_loss , outputs = self.unimo(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            pixel_values=pixel_values,
            aux_values=aux_values,
            rcnn_values=rcnn_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # print("kl_loss = ",kl_loss.shape)
        sequence_output = outputs[0]
        # print("sequence outputs = ",sequence_output.shape)
        prediction_scores = self.cls(sequence_output)
        # print("Prediction scores = ",prediction_scores.shape)
        # total_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss + kl_loss
        else:
            total_loss = kl_loss
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            # return ((masked_lm_loss, kl_loss),) + output if masked_lm_loss is not None else output
            return (masked_lm_loss,kl_loss) + output if masked_lm_loss is not None else output
        #print("Exiting forward of UnimoForMaskedLM of modelling_unimo")
        
        return MaskedLMOutput(
            loss=total_loss,
            # kl_loss = kl_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def get_output_embeddings(self):
        #print("Executing get_output_embeddings of UnimoForMaskedLM of modelling_unimo")
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        #print("Executing set_output_embeddings of UnimoForMaskedLM of modelling_unimo")
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        #print("Executing tie_weights of UnimoForMaskedLM of modelling_unimo")
        output_embeddings = self.get_output_embeddings()
        self._tie_or_clone_weights(output_embeddings, self.unimo.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        #print("Executing _tie_or_clone_weights of UnimoForMaskedLM of modelling_unimo")
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens):
        #print("Executing resize_token_embeddings of UnimoForMaskedLM of modelling_unimo")
        self.unimo.resize_token_embeddings(new_num_tokens)
        self.tie_weights()


class NeuralPrior(nn.Module):
    """
    Neural Prior is used to generate the visual prompt (patch).
    It takes a random noise as input and outputs an RGB patch.
    """
    def __init__(self, input_size):
        super(NeuralPrior, self).__init__()
        # Simple U-Net-like structure for generating the prompt
        self.unet = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1), nn.Sigmoid()  # Output is a normalized RGB patch
        )

    def forward(self, noise):
        return self.unet(noise)

class VisualPrompt(nn.Module):
    def __init__(self, prompt_size, image_size):
        super().__init__()
        self.prompt_size = prompt_size
        self.image_size = image_size
        self.neural_prior = NeuralPrior(input_size=prompt_size)  # Neural Prior for generating the prompt

    def forward(self, images):
        batch_size, channels, height, width = images.size()  # Get dimensions of the image tensor
        noise = torch.randn(batch_size, 3, self.prompt_size, self.prompt_size, device=images.device)  # Random noise

        # Generate the prompt using Neural Prior
        prompt = self.neural_prior(noise)

        # Ensure that the prompt can fit within the image dimensions
        if self.prompt_size > height or self.prompt_size > width:
            raise ValueError(f"Prompt size {self.prompt_size} is too large for the image size {height}x{width}")

        for i in range(batch_size):
            # Randomly select position to insert prompt ensuring it fits within the image
            x_pos = torch.randint(0, height - self.prompt_size, (1,)).item()
            y_pos = torch.randint(0, width - self.prompt_size, (1,)).item()

            # Insert the prompt at the chosen position
            images[i, :, x_pos:x_pos + self.prompt_size, y_pos:y_pos + self.prompt_size] = prompt[i]

        return images

def create_gaussian_target_map(patch_position, hidden_size, seq_length):
    """
    Creates a Gaussian target map centered around the patch position in token space.
    """
    gaussian_map = torch.zeros((seq_length, hidden_size))
    x_center, y_center = patch_position
    sigma = hidden_size / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))

    for i in range(seq_length):
        for j in range(hidden_size):
            gaussian_map[i, j] = torch.exp(-((i - x_center) ** 2 + (j - y_center) ** 2) / (2 * sigma ** 2))

    return gaussian_map

def compute_kl_loss(attention_weights, target_map):
    """
    Compute KL-Divergence between attention weights and the Gaussian target map.
    """
    attention_probs = F.softmax(attention_weights, dim=-1)
    target_probs = F.softmax(target_map, dim=-1)
    loss = F.kl_div(attention_probs.log(), target_probs, reduction='batchmean')
    return loss


class UnimoModel(nn.Module):
    def __init__(self, vision_config, text_config, add_pooling_layer=True, prompt_size=32):
        super(UnimoModel, self).__init__()
        # Vision model setup
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # Visual prompt integration with Neural Prior
        self.visual_prompt = VisualPrompt(prompt_size, vision_config.image_size)

        # Text model setup
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # Encoder combining vision and text
        self.encoder = UnimoEncoder(vision_config, text_config)

        self.device = vision_config.device

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                pixel_values=None, aux_values=None, rcnn_values=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, prompt_loss=True):
        
        # Visual prompt applied to image data using Neural Prior
        pixel_values = self.visual_prompt(pixel_values)

        # Vision embeddings
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # Text embeddings
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=self.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)

        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, self.device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)

        text_embedding_output = self.text_embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        # Encoder combining vision and text
        encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # If attention values are not available, use the last vision hidden state or average hidden states
        if output_attentions is None:
            # Use the final hidden state or average hidden states for KL loss
            vision_hidden_states = encoder_outputs.hidden_states  # all_vision_hidden_states
            if isinstance(vision_hidden_states, tuple):
                # Average all hidden states from all layers if available
                averaged_hidden_states = torch.mean(torch.stack(vision_hidden_states), dim=0)
            else:
                # Or simply use the last hidden state
                averaged_hidden_states = vision_hidden_states
        
        # Compute Gaussian map using the hidden states
        cls_token_position = (0, 0)  # You can set this to other positions
        gaussian_target_map = create_gaussian_target_map(cls_token_position, hidden_size=averaged_hidden_states.size(-1), seq_length=averaged_hidden_states.size(1))

        # Compute KL divergence loss
        if prompt_loss:
            kl_div_loss = compute_kl_loss(averaged_hidden_states, gaussian_target_map)
        else:
            kl_div_loss = 0  # If not applying the visual prompt loss

        # Return the encoder outputs and the loss for the visual prompt (if any)
        if return_dict:
            return {"encoder_outputs": encoder_outputs, "kl_loss": kl_div_loss}
        else:
            return encoder_outputs, kl_div_loss
