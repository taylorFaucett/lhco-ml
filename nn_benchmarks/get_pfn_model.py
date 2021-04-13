def get_pfn_model(network, tp):
    model = eval(network)(
                    input_dim=input_dim,
                    Phi_sizes= (tp["phi_size"], tp["phi_size"], tp["phi_size"]), 
                    F_sizes= (tp["f_size"], tp["f_size"], tp["f_size"]),
                    Phi_acts="relu", 
                    F_acts = "relu",
                    Phi_k_inits="glorot_normal",
                    F_k_inits="glorot_normal",
                    latent_dropout=tp["latent_dropout"],
                    F_dropouts=tp["f_dropout"],
                    mask_val = 0,
                    loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(lr=tp["learning_rate"]),
                    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
                    output_act="softmax",
                    summary=False
                )  
    return model
