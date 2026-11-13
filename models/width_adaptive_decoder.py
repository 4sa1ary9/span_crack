from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1,
                 padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                      padding=padding, bias=False),
            make_group_norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      groups=in_ch, bias=False),
            make_group_norm(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            make_group_norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CoordinateAttention(nn.Module):
    """Lightweight coordinate attention for thin-structure localisation."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm = make_group_norm(hidden, max_groups=4)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(-1, -2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.norm(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w.transpose(-1, -2)))
        return x * a_h * a_w


class MultiScaleContextBlock(nn.Module):
    """Parallel depthwise context aggregation for variable crack widths."""

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                          groups=channels, bias=False),
                make_group_norm(channels),
                nn.GELU(),
            )
            for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (len(dilations) + 1), channels, 1,
                      bias=False),
            make_group_norm(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x] + [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# Width prior estimator
# ---------------------------------------------------------------------------

class WidthPriorHead(nn.Module):
    """Predict a 3-class crack-width logit map from mid-level encoder features.

    Classes: thin (<= thin_max px by default 2), medium, wide.

    Returns raw logits (B, 3, H, W). Softmax is applied by the caller where a
    probability distribution is needed (fusion), so cross-entropy can consume
    the logits directly. Applying softmax here and then passing the result to
    cross-entropy double-normalises and weakens the gradient.

    Args:
        s2_ch: Stage 2 channels (64 for SegFormer-B0).
        s3_ch: Stage 3 channels (160 for SegFormer-B0).
        hidden: internal channel count.
    """

    def __init__(self, s2_ch: int = 64, s3_ch: int = 160, hidden: int = 64):
        super().__init__()
        self.proj = ConvGNAct(s2_ch + s3_ch, hidden, kernel_size=1)
        self.head = nn.Conv2d(hidden, 3, kernel_size=1)

    def forward(self, stage2: torch.Tensor, stage3: torch.Tensor,
                target_size: tuple[int, int]) -> torch.Tensor:
        """Returns (B, 3, H, W) raw width-class logits at *target_size*."""
        s2 = F.interpolate(stage2, size=target_size, mode='bilinear',
                           align_corners=False)
        s3 = F.interpolate(stage3, size=target_size, mode='bilinear',
                           align_corners=False)
        fused = self.proj(torch.cat([s2, s3], dim=1))
        return self.head(fused)


# ---------------------------------------------------------------------------
# Width-adaptive fusion decoder
# ---------------------------------------------------------------------------

class WidthAdaptiveFusionDecoder(nn.Module):
    """Decoder that fuses encoder stages with spatially-varying weights.

    A WidthPriorHead predicts a per-pixel crack-width class map. That map,
    together with the four projected encoder stages, feeds a lightweight
    spatial weight predictor that replaces the global scalar fusion weights
    of the original decoder.

    Args:
        s1_ch .. s4_ch: encoder stage channel counts.
        decoder_dim: shared projection dimension (default 64).
        fusion_mode: ``"spatial"`` (default) or ``"uniform"`` (ablation).
        width_mode: how the width prior enters fusion and supervision.
            ``"full"`` (default): the head runs, its softmax feeds fusion, and
            cross-entropy supervises the head. ``"learnable"``: no head;
            fusion consumes a single learnable 3-way softmax vector (W2).
            ``"constant"``: no head; fusion consumes a fixed 1/3 placeholder
            (W1). ``"aux_only"``: the head runs and is supervised, but fusion
            still uses the fixed 1/3 placeholder (W3). ``"none"``: no width
            branch at all; only meaningful with ``fusion_mode="uniform"``.
        use_channel_spatial_gate, use_coord_attention, use_ms_context:
            toggle post-fusion blocks (all default True).
        context_dilations: dilations for MultiScaleContextBlock.
    """

    def __init__(
        self,
        s1_ch: int = 32,
        s2_ch: int = 64,
        s3_ch: int = 160,
        s4_ch: int = 256,
        decoder_dim: int = 64,
        fusion_mode: str = "spatial",
        width_mode: str = "full",
        width_fusion_coupling: str = "concat",
        use_channel_spatial_gate: bool = True,
        use_coord_attention: bool = True,
        use_ms_context: bool = True,
        context_dilations: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        if fusion_mode not in {"spatial", "uniform"}:
            raise ValueError(
                f"fusion_mode must be 'spatial' or 'uniform', got '{fusion_mode}'"
            )
        if width_mode not in {"full", "learnable", "constant", "aux_only", "none"}:
            raise ValueError(
                f"width_mode must be one of full/learnable/constant/aux_only/none, "
                f"got '{width_mode}'"
            )
        if width_fusion_coupling not in {"concat", "routing"}:
            raise ValueError(
                f"width_fusion_coupling must be 'concat' or 'routing', "
                f"got '{width_fusion_coupling}'"
            )
        if fusion_mode == "uniform" and width_mode != "none":
            raise ValueError(
                "fusion_mode='uniform' requires width_mode='none' (no width branch)."
            )
        if fusion_mode == "spatial" and width_mode == "none":
            raise ValueError(
                "fusion_mode='spatial' cannot use width_mode='none'; use 'constant', "
                "'learnable', 'aux_only', or 'full'."
            )

        self.fusion_mode = fusion_mode
        self.width_mode = width_mode
        self.width_fusion_coupling = width_fusion_coupling
        self.decoder_dim = decoder_dim
        self.use_channel_spatial_gate = use_channel_spatial_gate
        self.use_coord_attention = use_coord_attention
        self.use_ms_context = use_ms_context

        # Stage projections
        self.s1_proj = ConvGNAct(s1_ch, decoder_dim, kernel_size=1)
        self.s2_proj = ConvGNAct(s2_ch, decoder_dim, kernel_size=1)
        self.s3_proj = ConvGNAct(s3_ch, decoder_dim, kernel_size=1)
        self.s4_proj = ConvGNAct(s4_ch, decoder_dim, kernel_size=1)

        # Width prior head: built only when a predicted width map is needed.
        self.has_width_head = width_mode in {"full", "aux_only"}
        if self.has_width_head:
            self.width_prior = WidthPriorHead(s2_ch, s3_ch, decoder_dim)
        # W2: one learnable 3-vector, softmaxed and broadcast to every pixel.
        if width_mode == "learnable":
            self.learnable_width = nn.Parameter(torch.zeros(1, 3, 1, 1))

        # Spatial fusion weight predictor
        if fusion_mode == "spatial":
            if width_fusion_coupling == "concat":
                # Baseline: width prior is concatenated with the 4 stages.
                spatial_in = decoder_dim * 4 + 3  # 4 stages + 3 width classes
            else:
                # routing: width does NOT enter the big conv (which drowns it);
                # it feeds a separate explicit routing path instead.
                spatial_in = decoder_dim * 4
            self.spatial_weight = nn.Sequential(
                nn.Conv2d(spatial_in, decoder_dim, 3, padding=1,
                          bias=False),
                make_group_norm(decoder_dim),
                nn.GELU(),
                nn.Conv2d(decoder_dim, 4, 1, bias=True),
            )
            if width_fusion_coupling == "routing":
                # Explicit width->scale routing: a learnable 3x4 matrix maps the
                # per-pixel width distribution to an additive bias on the 4
                # scale-fusion logits, so the width prior has a direct,
                # non-drownable path to fusion.
                #
                # Init note: width_route starts at small-random (std 0.01) and
                # route_scale starts at 1.0. Do NOT initialise both to zero:
                # that makes the routing bias zero AND kills the gradient to
                # both parameters (route_scale grad = route = 0, width_route
                # grad = route_scale * ... = 0), so the path would never learn.
                # Small-random weights keep the initial bias negligible
                # (~0.01 * 0.33) while letting gradients flow from step 1.
                self.width_route = nn.Linear(3, 4, bias=False)
                nn.init.normal_(self.width_route.weight, std=0.01)
                self.route_scale = nn.Parameter(torch.ones(1))

        # Post-fusion blocks
        if use_ms_context:
            self.ms_context = MultiScaleContextBlock(decoder_dim, context_dilations)
        if use_coord_attention:
            self.coord_attention = CoordinateAttention(decoder_dim)
        if use_channel_spatial_gate:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(decoder_dim, max(decoder_dim // 4, 8), 1, bias=False),
                nn.GELU(),
                nn.Conv2d(max(decoder_dim // 4, 8), decoder_dim, 1, bias=True),
                nn.Sigmoid(),
            )
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1,
                          groups=decoder_dim, bias=False),
                make_group_norm(decoder_dim),
                nn.GELU(),
                nn.Conv2d(decoder_dim, 1, 1, bias=True),
                nn.Sigmoid(),
            )

        # Refinement and prediction
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(decoder_dim, decoder_dim),
            DepthwiseSeparableConv(decoder_dim, decoder_dim),
        )
        self.mask_head = nn.Sequential(
            DepthwiseSeparableConv(decoder_dim, decoder_dim),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)

        if self.fusion_mode == "spatial":
            # Bias spatial weight predictor so weights start near-uniform
            nn.init.normal_(self.spatial_weight[-1].weight, std=0.01)
            nn.init.constant_(self.spatial_weight[-1].bias, 0.0)

    def forward(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(mask_logits, width_logits)``, or
        ``(mask_logits, decoder_features, width_logits)`` if *return_features*
        is True. ``width_logits`` are raw 3-class logits; the fusion internally
        uses their softmax, and training applies cross-entropy to the logits."""
        target_size = stage1.shape[-2:]

        # Project and align
        z1 = self.s1_proj(stage1)
        z2 = F.interpolate(self.s2_proj(stage2), size=target_size,
                           mode='bilinear', align_corners=False)
        z3 = F.interpolate(self.s3_proj(stage3), size=target_size,
                           mode='bilinear', align_corners=False)
        z4 = F.interpolate(self.s4_proj(stage4), size=target_size,
                           mode='bilinear', align_corners=False)
        branches = [z1, z2, z3, z4]

        # Width branch. Only "full" and "aux_only" run the head; "learnable"
        # uses a single learnable vector; "constant" uses a fixed 1/3 map.
        # Fusion always consumes a probability map of shape (B, 3, H, W); the
        # raw logits are returned when a head exists so cross-entropy can use
        # them directly.
        width_logits = None
        if self.width_mode in {"full", "aux_only"}:
            width_logits = self.width_prior(stage2, stage3, target_size)
            width_prob = F.softmax(width_logits, dim=1)
        elif self.width_mode == "learnable":
            width_prob = F.softmax(self.learnable_width, dim=1).expand(
                branches[0].shape[0], 3, *target_size
            )
        elif self.width_mode == "constant":
            width_prob = torch.full(
                (branches[0].shape[0], 3, *target_size),
                1.0 / 3.0,
                device=branches[0].device,
                dtype=branches[0].dtype,
            )
        else:  # "none" — uniform fusion path, no width branch
            width_prob = None

        if self.width_mode == "aux_only":
            # Head is supervised but its prediction must NOT drive fusion.
            width_prob = torch.full(
                (branches[0].shape[0], 3, *target_size),
                1.0 / 3.0,
                device=branches[0].device,
                dtype=branches[0].dtype,
            )

        # Fusion
        if self.fusion_mode == "spatial":
            if self.width_fusion_coupling == "concat":
                spatial_in = torch.cat(branches + [width_prob], dim=1)
                logits = self.spatial_weight(spatial_in)  # (B,4,H,W)
            else:
                # routing: base spatial logits from features only, plus an
                # explicit additive width->scale bias so the prior cannot be
                # drowned out by the feature channels.
                logits = self.spatial_weight(torch.cat(branches, dim=1))
                # width_prob (B,3,H,W) -> route (B,4,H,W)
                b, _, h, w = width_prob.shape
                route = self.width_route(
                    width_prob.permute(0, 2, 3, 1).reshape(-1, 3)
                ).reshape(b, h, w, 4).permute(0, 3, 1, 2)
                logits = logits + self.route_scale * route
            weights = F.softmax(logits, dim=1)  # (B,4,H,W)
            fused = sum(weights[:, i:i + 1] * branches[i] for i in range(4))
        else:
            fused = torch.stack(branches, dim=0).mean(dim=0)

        # Post-fusion
        if self.use_ms_context:
            fused = self.ms_context(fused)
        if self.use_coord_attention:
            fused = self.coord_attention(fused)
        if self.use_channel_spatial_gate:
            fused = fused * self.channel_gate(fused)
            fused = fused * (1.0 + self.spatial_gate(fused))

        features = self.refine(fused)
        logits = self.mask_head(features)

        if return_features:
            return logits, features, width_logits
        return logits, width_logits

    def export_fusion_weights(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (fusion_width_prob, fusion_weights) for diagnostics.

        ``fusion_width_prob`` (B, 3, H, W) is exactly the width map that drives
        fusion — i.e. the 1/3 placeholder for ``constant`` and ``aux_only``,
        the learnable softmax for ``learnable``, and the head softmax for
        ``full``. ``fusion_weights`` is (B, 4, H, W). Offline analysis only; not
        on the training or default inference path. To inspect the supervised
        head prediction under ``aux_only`` call the head directly.
        """
        target_size = stage1.shape[-2:]
        z1 = self.s1_proj(stage1)
        z2 = F.interpolate(self.s2_proj(stage2), size=target_size,
                           mode="bilinear", align_corners=False)
        z3 = F.interpolate(self.s3_proj(stage3), size=target_size,
                           mode="bilinear", align_corners=False)
        z4 = F.interpolate(self.s4_proj(stage4), size=target_size,
                           mode="bilinear", align_corners=False)
        branches = [z1, z2, z3, z4]

        if self.width_mode == "full":
            width_prob = F.softmax(
                self.width_prior(stage2, stage3, target_size), dim=1
            )
        elif self.width_mode == "learnable":
            width_prob = F.softmax(self.learnable_width, dim=1).expand(
                z1.shape[0], 3, *target_size
            )
        else:  # constant, aux_only: fusion sees the fixed 1/3 placeholder
            width_prob = torch.full(
                (z1.shape[0], 3, *target_size), 1.0 / 3.0,
                device=z1.device, dtype=z1.dtype,
            )
        if self.fusion_mode == "spatial":
            if self.width_fusion_coupling == "concat":
                logits = self.spatial_weight(torch.cat(branches + [width_prob], dim=1))
            else:
                logits = self.spatial_weight(torch.cat(branches, dim=1))
                b, _, h, w = width_prob.shape
                route = self.width_route(
                    width_prob.permute(0, 2, 3, 1).reshape(-1, 3)
                ).reshape(b, h, w, 4).permute(0, 3, 1, 2)
                logits = logits + self.route_scale * route
            weights = F.softmax(logits, dim=1)
        else:
            weights = torch.full(
                (z1.shape[0], 4, *target_size), 0.25,
                device=z1.device, dtype=z1.dtype,
            )
        return width_prob, weights
