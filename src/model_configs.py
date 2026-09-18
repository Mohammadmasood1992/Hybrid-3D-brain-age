"""
Architecture configurations compared in this project.

Two families of 10 total models are studied:

  A-E  Hybrid CNN + Transformer, all with the same parameter budget, varying
       only how many spatial tokens the transformer attends over (2 to 392).
       This is the main independent variable of the study: does giving the
       transformer more (finer-grained) tokens improve age prediction and
       change the accuracy/sensitivity trade-off on the ADHD holdout group?
  F-J  Plain 3D CNNs of varying depth and width, included as a baseline
       without any attention mechanism.
"""

from src.models import create_modular_cnn, create_modular_hybrid

DROPOUT = 0.1
EMBED_DIM = 128

TRANSFORMER_CONFIG = [
    {"num_heads": 4, "dropout": 0, "mlp_ratio": 2},
    {"num_heads": 4, "dropout": 0, "mlp_ratio": 2},
]
HEAD_STANDARD = [{"units": 128, "dropout": 0}, {"units": 64, "dropout": 0}]
HEAD_WIDE = [{"units": 512, "dropout": 0}, {"units": 256, "dropout": 0}, {"units": 64, "dropout": 0}]

# ---- Hybrid CNN+Transformer: token-count axis (A-E) -----------------------
CNN_5BLK_5POOL = [  # 1 x 1 x 2  =   2 tokens
    {"filters": 16, "dropout": DROPOUT, "use_pool": True},
    {"filters": 32, "dropout": DROPOUT, "use_pool": True},
    {"filters": 64, "dropout": DROPOUT, "use_pool": True},
    {"filters": 128, "dropout": DROPOUT, "use_pool": True},
    {"filters": 256, "dropout": DROPOUT, "use_pool": True},
]
CNN_5BLK_4POOL = [  # 3 x 3 x 4  =  36 tokens
    {"filters": 16, "dropout": DROPOUT, "use_pool": True},
    {"filters": 32, "dropout": DROPOUT, "use_pool": True},
    {"filters": 64, "dropout": DROPOUT, "use_pool": True},
    {"filters": 128, "dropout": DROPOUT, "use_pool": True},
    {"filters": 256, "dropout": DROPOUT, "use_pool": False},
]
CNN_5BLK_3POOL = [  # 7 x 7 x 8  = 392 tokens
    {"filters": 16, "dropout": DROPOUT, "use_pool": True},
    {"filters": 32, "dropout": DROPOUT, "use_pool": True},
    {"filters": 64, "dropout": DROPOUT, "use_pool": True},
    {"filters": 128, "dropout": DROPOUT, "use_pool": False},
    {"filters": 256, "dropout": DROPOUT, "use_pool": False},
]
CNN_4BLK_4POOL = [  # 3 x 3 x 4  =  36 tokens, one fewer conv block than B
    {"filters": 16, "dropout": DROPOUT, "use_pool": True},
    {"filters": 32, "dropout": DROPOUT, "use_pool": True},
    {"filters": 64, "dropout": DROPOUT, "use_pool": True},
    {"filters": 128, "dropout": DROPOUT, "use_pool": True},
]
CNN_3BLK_3POOL = [  # 7 x 7 x 8  = 392 tokens, one fewer conv block than C
    {"filters": 16, "dropout": DROPOUT, "use_pool": True},
    {"filters": 32, "dropout": DROPOUT, "use_pool": True},
    {"filters": 64, "dropout": DROPOUT, "use_pool": True},
]

# ---- Plain CNN baselines (F-J) ---------------------------------------------
# Note: ModularCNN3D uses a single conv per block, vs. the double conv used by
# the hybrid's CNN extractor above, so parameter counts are not matched
# across the two families -- see get_model_info()['n_params'] per model.
CNN_5BLK_NOATTN = [{"filters": f, "dropout": DROPOUT} for f in (16, 32, 64, 128, 256)]
CNN_6BLK_NOATTN = [{"filters": f, "dropout": DROPOUT} for f in (16, 32, 64, 128, 256)] + [
    {"filters": 256, "dropout": DROPOUT, "pool_size": None}
]
CNN_4BLK_PURE = [{"filters": f, "dropout": DROPOUT} for f in (16, 32, 64, 128)]
CNN_6BLK_PURE = [{"filters": f, "dropout": DROPOUT} for f in (16, 32, 64, 128, 256, 512)]
CNN_SFCN = [{"filters": f, "dropout": DROPOUT} for f in (32, 64, 128, 256, 256)] + [
    {"filters": 64, "dropout": DROPOUT, "kernel_size": 1, "pool_size": None}
]

# Builder functions, so every random seed starts each model from fresh weights.
MODEL_BUILDERS = {
    "A_5blk_2tok": lambda: create_modular_hybrid(CNN_5BLK_5POOL, TRANSFORMER_CONFIG, HEAD_STANDARD, embed_dim=EMBED_DIM),
    "B_5blk_36tok": lambda: create_modular_hybrid(CNN_5BLK_4POOL, TRANSFORMER_CONFIG, HEAD_STANDARD, embed_dim=EMBED_DIM),
    "C_5blk_392tok": lambda: create_modular_hybrid(CNN_5BLK_3POOL, TRANSFORMER_CONFIG, HEAD_STANDARD, embed_dim=EMBED_DIM),
    "D_4blk_36tok": lambda: create_modular_hybrid(CNN_4BLK_4POOL, TRANSFORMER_CONFIG, HEAD_STANDARD, embed_dim=EMBED_DIM),
    "E_3blk_392tok": lambda: create_modular_hybrid(CNN_3BLK_3POOL, TRANSFORMER_CONFIG, HEAD_STANDARD, embed_dim=EMBED_DIM),
    "F_noattn_wide": lambda: create_modular_cnn(CNN_5BLK_NOATTN, HEAD_WIDE, default_fc_dropout=DROPOUT),
    "G_noattn_deep": lambda: create_modular_cnn(CNN_6BLK_NOATTN, HEAD_STANDARD, default_fc_dropout=DROPOUT),
    "H_cnn4_d01": lambda: create_modular_cnn(CNN_4BLK_PURE, HEAD_STANDARD, default_fc_dropout=DROPOUT),
    "I_cnn6_d01": lambda: create_modular_cnn(CNN_6BLK_PURE, HEAD_STANDARD, default_fc_dropout=DROPOUT),
    "J_sfcn": lambda: create_modular_cnn(CNN_SFCN, HEAD_STANDARD, default_fc_dropout=DROPOUT),
}

SEEDS = [0, 1, 2]
