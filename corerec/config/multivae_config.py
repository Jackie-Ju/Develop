config = dict(
    model=dict(architecture='MultiVAE',
               mlp_hidden_size=[600],
               latent_dimension=128,
               dropout_prob=0.5
               ),
    loss=dict(type='CEKLLoss',
              anneal_cap=0.2,
              total_anneal_steps=200000,
              ),
    scheduler=dict(type="cosine_annealing",
                   T_max=200),
)