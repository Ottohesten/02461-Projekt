import tensorflow as tf


# tf_Conv_Qnet = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(16, 5, activation="relu"),
#     tf.keras.layers.Conv2D(32, 3, activation="relu"),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(32*4, activation="relu"),
#     tf.keras.layers.Dense(4, activation="linear")
# ])

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# tf_Conv_Qnet.compile(optimizer=optimizer, loss="mse")
# # print(tf_Conv_Qnet.summary())


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 5, activation="relu"),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32*4, activation="relu"),
        tf.keras.layers.Dense(4, activation="linear")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])
    return model