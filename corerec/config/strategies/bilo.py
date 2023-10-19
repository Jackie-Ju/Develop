# Learning setting
config = dict(
    dss_args=dict(type="Bilo",
                  optimizer='lazy',
                  selection_type='user'),

    train_args=dict(gradient_norm=False),

    eval_args=dict(coreset_affect=True),
)
