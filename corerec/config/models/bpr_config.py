config = dict(
    model=dict(architecture='BPR',
               embedding_size=64
               ),
    loss=dict(type='BPRLoss',
              gamma=1e-10)
)