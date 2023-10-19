config = dict(
    model=dict(architecture='LightGCN',
               embedding_size=64,
               n_layers=2,
               ),

    loss=dict(type='GCNLoss',
              gamma=1e-10,
              norm=2,
              require_pow=False,
              reg_weight=1e-05  # float32 type: the weight decay for l2 normalization
              )
)