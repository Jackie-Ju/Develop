config = dict(is_reg=False,
              dataset=dict(name="ml100k",
                           # /mnt: "/mnt/recsys/zhengju/projects/data"
                           # others: "~/projects/data"
                           # ichec: "/ichec/work/ndcom016c/zhengju/data"
                           datadir="/mnt/recsys/zhengju/projects/data",
                           feature="dss",
                           type="rating",
                           num_negatives=1,
                           ratio={'train': 0.8, 'valid': 0.1, 'test': 0.1},
                           core_user=None),

              dataloader=dict(shuffle=True,
                              batch_size=512,
                              ),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),

              optimizer=dict(type="adam",
                             momentum=0.9,
                             lr=0.001,
                             weight_decay=5e-4,
                             nesterov=False),

              scheduler=dict(type=None,#"cosine_annealing",
                             T_max=200),

              core_args=dict(setting=None,
                             fraction=0.1,
                             repeat=1,
                             seed=None),

              train_args=dict(num_epochs=200,
                              device="cpu",
                              valid=True,
                              verbose=True,
                              results_dir='results/',
                              return_args=[],
                              ),

              eval_args=dict(eval_every=2,
                             top_k=20,
                             verbose=True,
                             coreset_affect=True
                             ),
              # local servers
              # "./saved"
              # sonic
              #"/home/people/16205656/scratch/saved/"

              # kay.ichec
              # "/ichec/work/ndcom016c/zhengju/saved/"

              # nmt
              # "/mnt/recsys/zhengju/projects/test_gradient/gradient_core/saved"
              early_stopping=dict(root="./saved",
                                  ),
              logging=dict(state="info")
              )