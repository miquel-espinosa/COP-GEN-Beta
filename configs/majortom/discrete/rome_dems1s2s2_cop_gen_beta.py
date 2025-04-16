import ml_collections
def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)
def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.pred = "noise_pred"
    config.z_shape = (4, 32, 32)
    config.autoencoder = d(pretrained_path="assets/stable-diffusion/autoencoder_kl_ema.pth")
    config.train = d(
        n_steps=500000,
        batch_size=8, # Increase to 512 for larger datasets
        mode="uncond",
        log_interval=2, # Increase to 100 for larger datasets
        eval_interval=2, # Increase to 1500 for larger datasets
        save_interval=2, # Increase to 1500 for larger datasets
        multi_modal=True,
    )
    config.optimizer = d(
        name="adamw",
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )
    config.lr_scheduler = d(name="customized", warmup_steps=5000)
    config.nnet = d(
        name="triffuser_multi_post_ln",
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        num_modalities=4,
        use_checkpoint=True,
    )
    config.dataset = d(
        name="majorTOM_tuples_256_features",
        paths=["data/majorTOM/rome/rome_thumbnail_npy/train/DEM_thumbnail",
               "data/majorTOM/rome/rome_thumbnail_npy/train/S1RTC_thumbnail",
               "data/majorTOM/rome/rome_thumbnail_npy/train/S2L1C_thumbnail",
               "data/majorTOM/rome/rome_thumbnail_npy/train/S2L2A_thumbnail"],
        cfg=False,
        p_uncond=0.1, # 0.15
    )
    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=50,  # the decoder is large
        algorithm="dpm_solver",
        cfg=True,
        scale=0.4,
        path="",
    )
    return config
