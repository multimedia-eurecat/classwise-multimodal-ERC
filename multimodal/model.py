import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='sum')  

    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        target = target * mask_

        # Clamp target away from 0 to avoid xlogy backward NaNs
        eps = 1e-8
        target = target.clamp_min(eps)

        # Re-normalize across classes (only where mask=1)
        Z = target.sum(dim=-1, keepdim=True).clamp_min(eps)
        target = target / Z

        den = mask.sum().clamp_min(1.0)
        return self.loss(log_pred * mask_, target) / den


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())  
        return loss

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb[:, :L, :]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        mask = mask.unsqueeze(1)

        # Intra-modal: same utterance, same modality
        if inputs_a.equal(inputs_b):
            x = self.layer_norm(inputs_b)
            context = self.self_attn(x, x, x, mask=mask)
            out = self.dropout(context) + inputs_b
        
        # Inter-modal: same utterance, different modality
        else:
            qm = self.layer_norm(inputs_b)
            kn = self.layer_norm(inputs_a)
            context = self.self_attn(kn, kn, qm, mask=mask)
            out = self.dropout(context) + inputs_b

        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):

        # Intra-modal: same utterance, same modality
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        
        # Inter-modal: same utterance, different modality
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep


class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep


class ModalDisentangler(nn.Module):
    """
    Orthogonal Modality Decomposer (OMD).

    Splits each enhanced modality representation h'_m (d-dimensional) into two
    orthogonal subspaces:
    """
    def __init__(self, d: int):
        super().__init__()
        assert d % 2 == 0, f"hidden_dim must be even for ModalDisentangler, got {d}"
        self.to_shared   = nn.Linear(d, d // 2, bias=False)
        self.to_specific = nn.Linear(d, d // 2, bias=False)
        self.from_shared = nn.Linear(d // 2, d,  bias=False)

    def forward(self, x):
        # x: (B, N, d)
        shared   = self.to_shared(x)       # (B, N, d//2) — goes to fusion gate
        specific = self.to_specific(x)     # (B, N, d//2) — preserved, not fused
        expanded = self.from_shared(shared) # (B, N, d)    — gate input
        return expanded, shared, specific

    @staticmethod
    def orthogonality_loss(shared, specific):
        """Penalise the dot product between shared and specific vectors."""
        return (shared * specific).sum(dim=-1).pow(2).mean()
    

class Reliability_GatedFusion(nn.Module):
    """
    Class-wise reliability gating. For each modality m, a small MLP predicts
    a C-dimensional reliability score from [z_m; p_m]. Weights are softmax over
    the 3 modality scores per class. Final logits are a weighted sum of per-modality
    logit heads.

    Args:
        hidden_dim: dimension of z_m representations
        n_classes:  number of emotion classes C
        gate_hidden: intermediate dim of the gate MLP
    """
    def __init__(self, hidden_dim: int, n_classes: int, gate_hidden: int = 128):
        super().__init__()
        in_dim = hidden_dim + n_classes   # z_m || p_m

        # Separate gate MLPs per modality
        self.gate_t = nn.Sequential(
            nn.Linear(in_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, n_classes)
        )
        self.gate_a = nn.Sequential(
            nn.Linear(in_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, n_classes)
        )
        self.gate_v = nn.Sequential(
            nn.Linear(in_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, n_classes)
        )


    def forward(self, z_t, z_a, z_v, logit_t, logit_a, logit_v):
        def p(logits):
            return F.softmax(F.normalize(logits, dim=-1), dim=-1)  # logits already detached

        p_t, p_a, p_v = p(logit_t), p(logit_a), p(logit_v)

        B, N, D = z_t.shape
        def gate_input(z, prob):
            return torch.cat([z.reshape(B*N, D), prob.reshape(B*N, -1)], dim=-1)

        r_t = self.gate_t(gate_input(z_t, p_t)).reshape(B, N, -1)  # (B, N, C)
        r_a = self.gate_a(gate_input(z_a, p_a)).reshape(B, N, -1)
        r_v = self.gate_v(gate_input(z_v, p_v)).reshape(B, N, -1)

        w = F.softmax(torch.stack([r_t, r_a, r_v], dim=-1), dim=-1)  # (B, N, C, 3)

        # Reuse unimodal logits
        logits_stack = torch.stack([logit_t, logit_a, logit_v], dim=-1)  # (B, N, C, 3)
        return (logits_stack * w).sum(-1)                                  # (B, N, C)
    

class Scalar_Reliability_GatedFusion(nn.Module):
    """
    Scalar reliability gating — the colleague's original proposal.
 
    For each modality m, a small MLP predicts a SCALAR reliability score
    from [z_m; p_m]. Weights are softmax over 3 scalars → class-agnostic.
    The fused representation z_fused = Σ w_m * z_m is passed to
    all_output_layer for classification, exactly like entropy/softmax modes.
 
    Args:
        hidden_dim:  dimension of z_m representations
        n_classes:   number of emotion classes C (only used for p_m input size)
        gate_hidden: intermediate dim of the gate MLP
    """
    def __init__(self, hidden_dim: int, n_classes: int, gate_hidden: int = 128):
        super().__init__()
        in_dim = hidden_dim + n_classes
 
        self.gate_t = nn.Sequential(
            nn.Linear(in_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 1)       
        )
        self.gate_a = nn.Sequential(
            nn.Linear(in_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )
        self.gate_v = nn.Sequential(
            nn.Linear(in_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )
 
    def forward(self, z_t, z_a, z_v, logit_t, logit_a, logit_v):
        """
        Args:
            z_*:     (B, N, D)  modality representations (full gradient)
            logit_*: (B, N, C)  unimodal logits (detached — only used for p_m)
        Returns:
            z_fused: (B, N, D)  weighted sum of representations
        """
        def p(logits):
            # Calibrate: equalise dynamic range, then softmax
            return F.softmax(F.normalize(logits, dim=-1), dim=-1)  # (B, N, C)
 
        p_t, p_a, p_v = p(logit_t), p(logit_a), p(logit_v)
 
        B, N, D = z_t.shape
 
        def gate_input(z, prob):
            return torch.cat([z.reshape(B * N, D),
                              prob.reshape(B * N, -1)], dim=-1)  # (B*N, D+C)
 
        # Scalar reliability scores: (B, N, 1)
        r_t = self.gate_t(gate_input(z_t, p_t)).reshape(B, N, 1)
        r_a = self.gate_a(gate_input(z_a, p_a)).reshape(B, N, 1)
        r_v = self.gate_v(gate_input(z_v, p_v)).reshape(B, N, 1)
 
        # Softmax
        w = F.softmax(torch.cat([r_t, r_a, r_v], dim=-1), dim=-1)
 
        # Fuse representations
        z_fused = (w[..., 0:1] * z_t +
                   w[..., 1:2] * z_a +
                   w[..., 2:3] * z_v)   # (B, N, D)
        return z_fused
    

class Transformer_Reliability_GatedFusion(nn.Module):
    """
    Simple cross-modal transformer reliability gate.

    It treats text/audio/visual as three modality tokens, lets a small
    Transformer attend across them, then predicts one scalar reliability
    weight per modality.

    This version is intentionally simple and close to the reference code,
    but supports zeroed modality dropout through present_mask.
    """
    def __init__(self, hidden_dim: int, n_classes: int,
                 gate_dim: int = 64, n_heads: int = 2,
                 n_layers: int = 1, dropout: float = 0.1):
        super().__init__()

        in_dim = hidden_dim + n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, gate_dim),
            nn.LayerNorm(gate_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gate_dim,
            nhead=n_heads,
            dim_feedforward=gate_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.score_head = nn.Linear(gate_dim, 1, bias=False)

    def forward(self, z_t, z_a, z_v,
                logit_t, logit_a, logit_v,
                present_mask=None):
        """
        Args:
            z_*:
                (B, N, D) modality representations.

            logit_*:
                (B, N, C) detached unimodal logits.

            present_mask:
                Optional tuple:
                    (keep_t, keep_a, keep_v)

                Example:
                    (True, False, True) means audio was dropped.

        Returns:
            z_fused:
                (B, N, D)
        """
        def p(logits):
            return F.softmax(F.normalize(logits, dim=-1), dim=-1)

        p_t = p(logit_t)
        p_a = p(logit_a)
        p_v = p(logit_v)

        B, N, D = z_t.shape

        def make_token(z, prob):
            return torch.cat([z, prob], dim=-1).reshape(B * N, -1)

        tokens = torch.stack([
            make_token(z_t, p_t),
            make_token(z_a, p_a),
            make_token(z_v, p_v),
        ], dim=1)  # (B*N, 3, D+C)

        tokens = self.input_proj(tokens)  # (B*N, 3, gate_dim)

        src_key_padding_mask = None

        if present_mask is not None:
            present = torch.tensor(
                [bool(x) for x in present_mask],
                dtype=torch.bool,
                device=z_t.device,
            )

            # Defensive fallback. Normally your dropout code already keeps one.
            if not torch.any(present):
                present[:] = True

            dropped = ~present

            if torch.any(dropped):
                src_key_padding_mask = dropped.unsqueeze(0).expand(B * N, -1)

        tokens = self.transformer(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B*N, 3, gate_dim)

        scores = self.score_head(tokens).squeeze(-1)  # (B*N, 3)

        if present_mask is not None:
            present = torch.tensor(
                [bool(x) for x in present_mask],
                dtype=torch.bool,
                device=z_t.device,
            )

            if not torch.any(present):
                present[:] = True

            dropped = ~present

            if torch.any(dropped):
                scores = scores.masked_fill(
                    dropped.unsqueeze(0),
                    -1e4,
                )

        weights = F.softmax(scores, dim=-1).reshape(B, N, 3)

        z_fused = (
            weights[..., 0:1] * z_t +
            weights[..., 1:2] * z_a +
            weights[..., 2:3] * z_v
        )

        return z_fused
    
class LandmarkProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D_in)
        T, B, D = x.shape
        return self.net(x.reshape(T * B, D)).reshape(T, B, -1).permute(1, 0, 2)  # (B, T, hidden_dim)
    

class IntraVisualGate(nn.Module):
    def __init__(self, vit_dim: int, geo_dim: int,
                 hidden_dim: int, gate_hidden: int = 64,
                 dropout: float = 0.1, use_gate: bool = True):
        super().__init__()
        self.vit_dim = vit_dim
        self.geo_dim = geo_dim
        self.use_gate = use_gate

        self.proj_vit = nn.Conv1d(vit_dim, hidden_dim, kernel_size=1, bias=False)

        self.proj_geo = nn.Sequential(
            nn.LayerNorm(geo_dim),
            nn.Linear(geo_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 2),
            )
        else:
            self.gate = None

    def forward(self, visuf: torch.Tensor) -> torch.Tensor:
        """
        visuf: (T, B, D_visual), expected order [ViT, geometry]
        returns: (B, T, hidden_dim)
        """
        expected_dim = self.vit_dim + self.geo_dim
        assert visuf.shape[-1] == expected_dim, \
            f"Expected D_visual={expected_dim}, got {visuf.shape[-1]}"

        vit_raw = visuf[..., :self.vit_dim]
        geo_raw = visuf[..., self.vit_dim:]

        h_vit = self.proj_vit(
            vit_raw.permute(1, 2, 0)
        ).transpose(1, 2)

        T, B, _ = geo_raw.shape
        h_geo = self.proj_geo(
            geo_raw.reshape(T * B, -1)
        ).reshape(T, B, -1).permute(1, 0, 2)

        h_vit = F.layer_norm(h_vit, [h_vit.shape[-1]])
        h_geo = F.layer_norm(h_geo, [h_geo.shape[-1]])

        if self.use_gate:
            # Learned per-utterance routing between ViT and geometry
            B, T, D = h_vit.shape
            w = F.softmax(
                self.gate(
                    torch.cat([h_vit, h_geo], dim=-1).reshape(B * T, -1)
                ).reshape(B, T, 2),
                dim=-1,
            )                            # (B, T, 2)
            return w[..., 0:1] * h_vit + w[..., 1:2] * h_geo
        else:
            return 0.5 * h_vit + 0.5 * h_geo
    

class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, modal_dropout=0.0, dropout_strategy='input',
                 fusion_mode='softmax', con_temperature=0.07, use_disentangle=False,
                 use_intra_visual_gate=False, use_intra_visual_gate_learned=True, use_speaker_embeddings=True, 
                 use_landmark_mlp=False, gate_dim=64, circumplex_alpha=0.0, va_map=None,
                 circumplex_tau=0.5):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.con_temperature = con_temperature
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.modal_dropout = modal_dropout
        self.dropout_strategy = dropout_strategy
        self.fusion_mode = fusion_mode
        self.use_disentangle = use_disentangle
        self.use_speaker_embeddings = use_speaker_embeddings
        self.use_landmark_mlp = use_landmark_mlp
        self.use_intra_visual_gate = use_intra_visual_gate and (D_visual > 768)
        
        if self.use_intra_visual_gate:
            self.intra_visual_gate = IntraVisualGate(
                vit_dim=768, geo_dim=D_visual - 768, hidden_dim=hidden_dim,
                dropout=dropout, use_gate=use_intra_visual_gate_learned,
            )

        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)

        # Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        if use_landmark_mlp:
            self.visuf_input = LandmarkProjector(D_visual, hidden_dim, dropout)
        else:
            self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
                
        # Intra- and Inter-modal Transformers
        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        
        # Unimodal-level Gated Fusion
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        self.disent_t = ModalDisentangler(hidden_dim)
        self.disent_a = ModalDisentangler(hidden_dim)
        self.disent_v = ModalDisentangler(hidden_dim)
                                          
        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Reliability Gated Fusion
        self.reliability_gate = Reliability_GatedFusion(hidden_dim, n_classes)

        # Scalar Reliability Gate
        self.scalar_reliability_gate = Scalar_Reliability_GatedFusion(hidden_dim, n_classes)
        
        # Transformer Reliability Gate
        self.transformer_reliability_gate = Transformer_Reliability_GatedFusion(
            hidden_dim, n_classes,
            gate_dim=gate_dim,
        )

        # Emotion Classifier
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

        # ── Circumplex transition prior ────────────────────────────────────────
        # va_map: {class_idx: (valence, arousal)}, values normalised to [0, 1]
        self.circumplex_alpha = circumplex_alpha
        self.circumplex_tau   = circumplex_tau
        if va_map is not None:
            _va = torch.tensor(
                [va_map[i] for i in range(n_classes)], dtype=torch.float
            )  # (C, 2)
            self.register_buffer('va_coords', _va)
        else:
            self.register_buffer('va_coords', None)
    @staticmethod
    def _entropy_fusion(t, a, v, t_logits, a_logits, v_logits):
        """
        Weighted sum of t/a/v representations using inverse entropy of each
        modality's student classifier output as the weight.
        """
        def entropy(logits):
            p = F.softmax(logits.detach(), dim=-1)          # (B, N, C)       
            H = -(p * torch.log(p.clamp(min=1e-9))).sum(-1) # (B, N)
            return H
 
        H_t = entropy(t_logits)
        H_a = entropy(a_logits)
        H_v = entropy(v_logits)
 
        # Lower entropy --> higher weight
        H_stack = torch.stack([-H_t, -H_a, -H_v], dim=-1)   
        weights = F.softmax(H_stack, dim=-1)                                 
 
        fused = (weights[..., 0:1] * t +
                 weights[..., 1:2] * a +
                 weights[..., 2:3] * v)
                          
        return fused

    def _compute_circumplex_prior(self, all_final_out):
        """
        VA-geometry transition prior, applied specifically at predicted shift moments.

        Russell's circumplex organises emotion classes by their position in 2D
        valence-arousal space. Consecutive utterances tend to either stay in
        the same region (emotional inertia) or make a jump across the space.
        This prior adds a soft geometric constraint: when the model predicts a
        large shift, it nudges the logits toward classes that are close to where
        the conversation just was in VA space.

        Key design choices
        ──────────────────
        1. Soft previous distribution (not hard argmax)
           Uses softmax(logits_{i-1}) to estimate the previous VA position,
           so training and inference use the exact same computation path —
           no teacher forcing, no exposure bias.

        2. Expected VA position
           E[VA_{i-1}] = prev_prob @ va_coords maps the soft distribution
           to a continuous point in the 2D circumplex.

        3. Distance-based log-prior
           log_softmax(-dist / tau) — classes near E[VA_{i-1}] get higher
           log-probability. tau controls sharpness: small tau = sharp prior
           that strongly prefers the nearest class; large tau = flatter prior.

        4. Severity gating (the key ingredient)
           severity = ||E[VA_i] - E[VA_{i-1}]|| — how much the predicted VA
           position moved between consecutive utterances.
           Prior contribution = log_prior * severity.
           At stable moments (small severity): near-zero contribution.
           At shift moments (large severity): full geometric adjustment.
           This is what makes the prior help shifts without hurting stable.

        Returns: (B, N, C) log-prior, added to all_final_out with weight
                 circumplex_alpha in the forward pass.
        """
        B, N, C = all_final_out.shape

        # Soft predictions at each position — detached to avoid circular gradients
        curr_prob = F.softmax(all_final_out.detach(), dim=-1)    # (B, N, C)

        # Shift by 1: previous soft distribution
        # First utterance has no history → use uniform prior
        uniform   = curr_prob.new_full((B, 1, C), 1.0 / C)
        prev_prob = torch.cat([uniform, curr_prob[:, :-1, :]], dim=1)  # (B, N, C)

        # Expected VA positions in the [0,1]^2 circumplex
        E_va_prev = prev_prob @ self.va_coords   # (B, N, 2)
        E_va_curr = curr_prob @ self.va_coords   # (B, N, 2)

        # Euclidean distance from expected previous VA to every class centroid
        # va_coords: (C, 2) → broadcast to (B, N, C, 2)
        va_all        = self.va_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, C, 2)
        E_va_prev_exp = E_va_prev.unsqueeze(2)                    # (B, N, 1, 2)
        distances     = torch.norm(va_all - E_va_prev_exp, dim=-1)  # (B, N, C)

        # Log-prior: classes near the previous VA position are preferred
        log_prior = F.log_softmax(-distances / self.circumplex_tau, dim=-1)  # (B, N, C)

        # Severity: predicted VA velocity — activates the prior only at shift moments
        severity = torch.norm(E_va_curr - E_va_prev, dim=-1, keepdim=True)   # (B, N, 1)

        return log_prior * severity   # (B, N, C)

    @staticmethod
    def _supervised_contrastive_loss(h_t, h_a, h_v, labels, mask, temperature=0.07):
        """
        Supervised contrastive loss across modalities (Khosla et al., NeurIPS 2020).
 
        Forces text, audio and visual representations of the same emotion to
        cluster together in the shared space, regardless of which modality they
        came from.  This makes the representation space modality-invariant:
        when a modality is weak or dropped, the other two already occupy the
        same neighbourhood, so fusion degrades gracefully.
 
        Args:
            h_t, h_a, h_v : (B, N, d)  enhanced modality representations
                             (output of features_reduce_*, before last_gate)
            labels         : (B, N)     integer emotion class ids
            mask           : (B, N)     binary valid-utterance mask (1 = valid)
            temperature    : float      softmax temperature; lower → sharper
                             distribution, harder negatives.  0.07 is the
                             SimCLR/SupCon default; try 0.1–0.2 if loss diverges.
 
        Returns:
            scalar loss (0.0 if fewer than 2 valid utterances in the batch)
        """
        B, N, d = h_t.shape
 
        # 1. Normalize representation with L2 and flatten to utterance level → (B*N, d)
        def flat_norm(x):
            return F.normalize(x.reshape(B * N, d), dim=-1)
 
        ht = flat_norm(h_t)
        ha = flat_norm(h_a)
        hv = flat_norm(h_v)
 
        labs  = labels.reshape(B * N)       
        valid = mask.reshape(B * N).bool()  
 
        # 2. Drop padding positions, since dialogue batches usually contain padded utterances to 
        # make all dialogues the same length
        ht, ha, hv = ht[valid], ha[valid], hv[valid]
        labs = labs[valid]
        M = ht.shape[0]   # number of valid utterances in this batch
 
        if M < 2:
            return torch.tensor(0.0, device=h_t.device, requires_grad=True)
 
        # 3. Stack all 3 modalities as different "views"
        # Layout: [text_0..M-1 | audio_0..M-1 | visual_0..M-1]
        all_rep = torch.cat([ht, ha, hv], dim=0)   # (3M, d)
        all_lab = labs.repeat(3)                    # (3M,)
 
        # 4. Compute pairwise cosine similarities
        sim = torch.matmul(all_rep, all_rep.T) / temperature
 
        # 5. Remove self-comparisons
        eye = torch.eye(3 * M, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(eye, -1e9)
 
        # 6. Define positive pairs using labels
        pos_mask = (all_lab.unsqueeze(0) == all_lab.unsqueeze(1)) & ~eye
 
        # Anchors with at least one positive (rare emotions may have none)
        n_pos   = pos_mask.sum(dim=-1).float()   
        has_pos = n_pos > 0                      
 
        if not has_pos.any():
            return torch.tensor(0.0, device=h_t.device, requires_grad=True)
 
        # Log-softmax denominator = all non-self pairs
        log_prob = F.log_softmax(sim, dim=-1)    
 
        # Mean log-prob over positives per anchor, then mean over valid anchors
        per_anchor = -(log_prob * pos_mask).sum(dim=-1) / n_pos.clamp(min=1)
        loss = per_anchor[has_pos].mean()
        return loss


    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, labels=None, 
                shift_severity=None, force_drop_modalities=None, speaker_known_mask=None):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        device = spk_idx.device
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                x = min(x, origin_spk_idx[i].size(0))
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().to(device)
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                x = min(x, origin_spk_idx[i].size(0))
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().to(device)
        
        if self.use_speaker_embeddings:
            spk_embeddings = self.speaker_embeddings(spk_idx)
        else:
            spk_embeddings = torch.zeros(
                spk_idx.size(0),
                spk_idx.size(1),
                self.speaker_embeddings.embedding_dim,
                device=spk_idx.device,
                dtype=self.speaker_embeddings.weight.dtype,
            )

        if speaker_known_mask is not None:
            if speaker_known_mask.shape != spk_idx.shape:
                raise ValueError("speaker_known_mask must match the batch and utterance dimensions")
            if speaker_known_mask.dtype != torch.bool:
                raise ValueError("speaker_known_mask must be boolean")
            # Missing identity is no speaker signal, not a third identity or a
            # padded utterance. u_mask still retains this turn in attention.
            spk_embeddings = spk_embeddings.masked_fill(
                ~speaker_known_mask.to(device).unsqueeze(-1), 0.0
            )

        force_drop_modalities = set(force_drop_modalities or [])
        drop_t = drop_a = drop_v = False
        if len(force_drop_modalities) > 0:
            # Deterministic ablation for evaluation.
            drop_t = 'T' in force_drop_modalities
            drop_a = 'A' in force_drop_modalities
            drop_v = 'V' in force_drop_modalities

        elif self.training and self.modal_dropout > 0.0:
            # Stochastic modality dropout for training regularization.
            drop_t = torch.rand(1).item() < self.modal_dropout
            drop_a = torch.rand(1).item() < self.modal_dropout
            drop_v = torch.rand(1).item() < self.modal_dropout

            # Always keep at least one modality.
            if drop_t and drop_a and drop_v:
                keep = torch.randint(3, (1,)).item()
                drop_t = keep != 0
                drop_a = keep != 1
                drop_v = keep != 2

        # Input dropout
        if self.dropout_strategy == 'input':
            if drop_t:
                textf = torch.zeros_like(textf)
            if drop_a:
                acouf = torch.zeros_like(acouf)
            if drop_v:
                visuf = torch.zeros_like(visuf)

        # Temporal convolutional layers
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        if self.use_intra_visual_gate:
            visuf = self.intra_visual_gate(visuf)
        elif self.use_landmark_mlp:
            visuf = self.visuf_input(visuf)
        else:
            visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        # Intra- and Inter-modal Transformers
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)
        v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings)

        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings)

        # Unimodal-level Gated Fusion
        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.a_t_gate(a_t_transformer_out)
        v_t_transformer_out = self.v_t_gate(v_t_transformer_out)

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out)
        v_a_transformer_out = self.v_a_gate(v_a_transformer_out)

        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(t_v_transformer_out)
        a_v_transformer_out = self.a_v_gate(a_v_transformer_out)

        t_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))

        # Representation dropout
        if self.dropout_strategy == 'representation':
            if drop_t:
                t_transformer_out = torch.zeros_like(t_transformer_out)
            if drop_a:
                a_transformer_out = torch.zeros_like(a_transformer_out)
            if drop_v:
                v_transformer_out = torch.zeros_like(v_transformer_out)

        orth_loss = None
        if self.use_disentangle:
            t_gate, t_shared, t_spec = self.disent_t(t_transformer_out)
            a_gate, a_shared, a_spec = self.disent_a(a_transformer_out)
            v_gate, v_shared, v_spec = self.disent_v(v_transformer_out)
            orth_loss = (
                ModalDisentangler.orthogonality_loss(t_shared, t_spec) +
                ModalDisentangler.orthogonality_loss(a_shared, a_spec) +
                ModalDisentangler.orthogonality_loss(v_shared, v_spec)
            )
            # What enters last_gate: expanded shared representations
            t_for_gate, a_for_gate, v_for_gate = t_gate, a_gate, v_gate
            # What enters the contrastive loss: shared subspace
            t_for_con, a_for_con, v_for_con = t_shared, a_shared, v_shared
        else:
            t_for_gate = t_transformer_out
            a_for_gate = a_transformer_out
            v_for_gate = v_transformer_out
            t_for_con  = t_transformer_out
            a_for_con  = a_transformer_out
            v_for_con  = v_transformer_out


        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)

        # Multimodal-level Gated Fusion
        present_mask = (not drop_t, not drop_a, not drop_v)
        if self.fusion_mode == 'class_reliability':
            all_final_out = self.reliability_gate(
                t_for_gate, a_for_gate, v_for_gate,
                t_final_out.detach(), a_final_out.detach(), v_final_out.detach()
            )

        elif self.fusion_mode == 'scalar_reliability':
            all_transformer_out = self.scalar_reliability_gate(
                t_for_gate, a_for_gate, v_for_gate,
                t_final_out.detach(), a_final_out.detach(), v_final_out.detach()
            )
            all_final_out = self.all_output_layer(all_transformer_out)

        elif self.fusion_mode == 'transformer_reliability':
            all_transformer_out = self.transformer_reliability_gate(
                t_for_gate, a_for_gate, v_for_gate,
                t_final_out.detach(), a_final_out.detach(), v_final_out.detach(),
                present_mask=present_mask,
            )
            all_final_out = self.all_output_layer(all_transformer_out)

        else:
            if self.fusion_mode == 'entropy':
                all_transformer_out = self._entropy_fusion(
                    t_for_gate, a_for_gate, v_for_gate,
                    t_final_out, a_final_out, v_final_out,
                )
            else:
                all_transformer_out = self.last_gate(
                    t_for_gate, a_for_gate, v_for_gate
                )

            # Emotion Classifier
            all_final_out = self.all_output_layer(all_transformer_out)


        # Circumplex transition prior: bias logits toward VA-plausible transitions,
        # scaled by predicted shift severity so stable utterances are unaffected.
        if self.circumplex_alpha > 0 and self.va_coords is not None:
            prior = self._compute_circumplex_prior(all_final_out)
            all_final_out = all_final_out + self.circumplex_alpha * prior

        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out /self.temp, 2)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)

        if self.fusion_mode == 'class_reliability':
            kl_all_prob = kl_all_prob.detach()

        # Supervised contrastive loss
        con_loss = None
        if self.training and labels is not None and self.con_temperature > 0:
            con_loss = self._supervised_contrastive_loss(
                t_for_con, a_for_con, v_for_con,
                labels, u_mask, temperature=self.con_temperature,
            )

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob, con_loss, orth_loss
