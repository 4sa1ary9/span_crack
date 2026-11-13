from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from models.crack_selective_adapter import CrackSelectiveAdapter
from models.texture_suppression import TextureSuppressionGate
from models.width_adaptive_decoder import WidthAdaptiveFusionDecoder
from utils.project import resolve_project_path


MODEL_DIRS = {
    "b0": resolve_project_path("pretrained", "mit-b0"),
    "b1": resolve_project_path("pretrained", "mit-b1"),
    "b2": resolve_project_path("pretrained", "mit-b2"),
    "b3": resolve_project_path("pretrained", "mit-b3"),
}


class TCSSegFormer(nn.Module):
    """Texture-Crack-Structure SegFormer for pavement crack segmentation.

    Four optional inductive biases on a pretrained SegFormer-B0 backbone:

    - **Crack-Selective Adaptation**: lightweight residual adapters applied
      after each encoder stage to reshape features toward crack evidence.
    - **Texture Suppression**: a lightweight gate after Stage 1 that learns
      to suppress pavement texture before decoding.
    - **Width-Adaptive Fusion**: a decoder that predicts a crack-width prior
      and uses it for spatially-varying multi-scale fusion.
    - **Topology-Aware Supervision**: skeleton and boundary auxiliary heads
      trained with soft-clDice and path-continuity losses.

    Set *variant* to ``"pure_segformer"`` for the unmodified baseline.
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_type: str = "b0",
        variant: str = "tcs_seg",
        arch_config: dict | None = None,
    ):
        super().__init__()
        if model_type not in MODEL_DIRS:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.variant = variant
        self.arch_config = arch_config or {}
        decoder_config = {}
        if "segformer_decoder_dim" in self.arch_config:
            width = self.arch_config["segformer_decoder_dim"]
            if variant != "pure_segformer" or not isinstance(width, int) or isinstance(width, bool) or width <= 0:
                raise ValueError("segformer_decoder_dim must be a positive integer on pure_segformer")
            decoder_config["decoder_hidden_size"] = width

        local_model_path = MODEL_DIRS[model_type]
        if not local_model_path.exists():
            raise FileNotFoundError(
                f"Missing local pretrained model directory: {local_model_path}"
            )

        if pretrained:
            self.base_model = SegformerForSemanticSegmentation.from_pretrained(
                str(local_model_path),
                num_labels=1,
                ignore_mismatched_sizes=True,
                local_files_only=True,
                **decoder_config,
            )
        else:
            config = SegformerConfig.from_pretrained(
                str(local_model_path),
                num_labels=1,
                local_files_only=True,
                **decoder_config,
            )
            self.base_model = SegformerForSemanticSegmentation(config)

        hidden_sizes = self.base_model.config.hidden_sizes
        c1, c2, c3, c4 = hidden_sizes

        if self.variant not in {"pure_segformer", "tcs_seg"}:
            raise ValueError(
                f"Unsupported model variant: {self.variant}. "
                "Expected one of ['pure_segformer', 'tcs_seg']."
            )

        if self.variant == "tcs_seg":
            self._build_tcs_modules(c1, c2, c3, c4)

    # ------------------------------------------------------------------
    # TCS-Seg module construction
    # ------------------------------------------------------------------

    def _build_tcs_modules(self, c1: int, c2: int, c3: int, c4: int):
        cfg = self.arch_config
        decoder_dim = int(cfg.get("decoder_dim", 64))

        # Crack-selective adapters (one per encoder stage)
        self.use_crack_adapters = bool(cfg.get("use_crack_adapters", False))
        if self.use_crack_adapters:
            adapter_reduction = int(cfg.get("adapter_reduction", 4))
            adapter_init_scale = float(cfg.get("adapter_init_scale", 0.0))
            self.adapter_s1 = CrackSelectiveAdapter(
                channels=c1, reduction=adapter_reduction,
                init_scale=adapter_init_scale,
            )
            self.adapter_s2 = CrackSelectiveAdapter(
                channels=c2, reduction=adapter_reduction,
                init_scale=adapter_init_scale,
            )
            self.adapter_s3 = CrackSelectiveAdapter(
                channels=c3, reduction=adapter_reduction,
                init_scale=adapter_init_scale,
            )
            self.adapter_s4 = CrackSelectiveAdapter(
                channels=c4, reduction=adapter_reduction,
                init_scale=adapter_init_scale,
            )

        # Texture suppression gate (after Stage 1)
        self.use_tsm = bool(cfg.get("use_texture_suppression", True))
        if self.use_tsm:
            self.tsm_gate = TextureSuppressionGate(
                channels=c1,
                reduction=int(cfg.get("tsm_reduction", 4)),
            )

        # Width-adaptive fusion decoder
        self.use_waf = bool(cfg.get("use_width_adaptive_fusion", True))
        fusion_mode = "spatial" if self.use_waf else "uniform"
        # Width branch behaviour. When WAF is off there is no width branch.
        if self.use_waf:
            self.width_mode = str(cfg.get("width_mode", "full"))
        else:
            self.width_mode = "none"

        self.fusion_decoder = WidthAdaptiveFusionDecoder(
            s1_ch=c1, s2_ch=c2, s3_ch=c3, s4_ch=c4,
            decoder_dim=decoder_dim,
            fusion_mode=fusion_mode,
            width_mode=self.width_mode,
            width_fusion_coupling=str(cfg.get("width_fusion_coupling", "concat")),
            use_channel_spatial_gate=bool(cfg.get("use_channel_spatial_gate", True)),
            use_coord_attention=bool(cfg.get("use_coord_attention", True)),
            use_ms_context=bool(cfg.get("use_ms_context", True)),
            context_dilations=tuple(
                int(d) for d in cfg.get("context_dilations", [1, 2, 3])
            ),
        )

        # Topology auxiliary heads
        self.use_topology_aux_heads = bool(
            cfg.get("use_topology_aux_heads", True)
        )
        if self.use_topology_aux_heads:
            self.skeleton_head = nn.Sequential(
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1,
                          groups=decoder_dim, bias=False),
                nn.GELU(),
                nn.Conv2d(decoder_dim, 1, 1),
            )
            self.boundary_head = nn.Sequential(
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1,
                          groups=decoder_dim, bias=False),
                nn.GELU(),
                nn.Conv2d(decoder_dim, 1, 1),
            )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.base_model.segformer(
            pixel_values=x, output_hidden_states=True
        )
        return list(outputs.hidden_states)

    def _decode_segformer(
        self, hidden_states: list[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        logits = self.base_model.decode_head(hidden_states)
        return F.interpolate(logits, size=output_size,
                             mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def _forward_pure_segformer(
        self, hidden_states: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        return self._decode_segformer(hidden_states, output_size)

    def _forward_tcs_seg(
        self,
        hidden_states: list[torch.Tensor],
        output_size: tuple[int, int],
        return_aux: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        stage1, stage2, stage3, stage4 = hidden_states

        # Crack-selective adaptation: reshape each stage toward crack evidence
        # before any downstream processing.
        if self.use_crack_adapters:
            stage1 = self.adapter_s1(stage1)
            stage2 = self.adapter_s2(stage2)
            stage3 = self.adapter_s3(stage3)
            stage4 = self.adapter_s4(stage4)

        # Texture suppression: gate stage1 before it reaches the decoder.
        # Stage2-4 are computed from ungated stage1 (backbone is untouched).
        if self.use_tsm:
            stage1, _gate_map = self.tsm_gate(stage1)

        wants_aux = return_aux and self.use_topology_aux_heads
        decoder_out = self.fusion_decoder(
            stage1, stage2, stage3, stage4,
            return_features=wants_aux,
        )

        if wants_aux:
            mask_logits, decoder_features, width_logits = decoder_out
        else:
            mask_logits, width_logits = decoder_out
            decoder_features = None

        mask_logits = F.interpolate(
            mask_logits, size=output_size,
            mode="bilinear", align_corners=False,
        )

        if not return_aux:
            return mask_logits

        outputs = {
            "mask_logits": mask_logits,
        }
        if width_logits is not None:
            outputs["width_logits"] = width_logits
            outputs["width_prior"] = torch.softmax(width_logits, dim=1)

        if decoder_features is not None:
            outputs["skeleton_logits"] = F.interpolate(
                self.skeleton_head(decoder_features),
                size=output_size, mode="bilinear", align_corners=False,
            )
            outputs["boundary_logits"] = F.interpolate(
                self.boundary_head(decoder_features),
                size=output_size, mode="bilinear", align_corners=False,
            )

        return outputs

    def forward(
        self, x: torch.Tensor, return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        hidden_states = self._encode(x)

        if self.variant == "pure_segformer":
            logits = self._forward_pure_segformer(hidden_states, input_shape)
            if return_aux:
                return {"mask_logits": logits}
            return logits

        return self._forward_tcs_seg(
            hidden_states, input_shape, return_aux=return_aux,
        )
