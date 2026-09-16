"""Quick smoke check for stage-1 modules (not part of the test suite)."""

import torch

from tdarts import config as C
from tdarts.backbone import TemporalBackbone
from tdarts.temporal_ops import build_all_candidates, effective_rf

print("BASE_KERNEL :", C.BASE_KERNEL)
print("RF_SPACE    :", C.RF_SPACE)
print("DILATIONS   :", C.DILATIONS)
print("OPERATORS   :", C.OPERATORS)
print("NUM_FEAT    :", C.NUM_FEAT)
print("band_groups :", C.band_groups())

x = torch.randn(2, C.IN_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
print("\ninput:", tuple(x.shape))

cands = build_all_candidates(use_norm=False)
print("candidates:", len(cands))
for key, cand in cands.items():
    band, op_name, rf = key
    y = cand(x)
    assert y.shape == (2, C.PATH_CHANNELS, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS), (key, y.shape)
    assert cand.effective_rf == rf, (key, cand.effective_rf, rf)
    if rf == 57 and band == "Low":
        print("  ", cand.describe(), "| layers:", len(cand.op._layers))

print("\nall 48 shapes + RF ok")

cands_norm = build_all_candidates(use_norm=True)
y = cands_norm[("Low", "dilated", 57)](x)
print("with candidate norm:", tuple(y.shape))

# backbone
backbone = TemporalBackbone()
print("\nbackbone in_channels:", backbone.in_channels, "feature_dim:", backbone.feature_dim)
feat_in = torch.randn(2, C.NUM_FEAT, C.NUM_ELECTRODES, C.NUM_TIMEPOINTS)
logits, features = backbone(feat_in)
print("backbone input :", tuple(feat_in.shape))
print("scb out        :", tuple(backbone.scb(feat_in).shape))
print("logits         :", tuple(logits.shape))
print("features       :", tuple(features.shape))

# two paths x 4 RFs of one band concatenated = 12 channels = FBNAS per-band width
two_paths = torch.cat([cands[("Low", "dilated", 57)](x), cands[("Low", "lkdw", 113)](x)], dim=1)
print("\ntwo-path Low shape:", tuple(two_paths.shape))

print("\nOK")
