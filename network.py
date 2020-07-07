import tensorflow as tf
import tensorflow.contrib.layers as layers

def network(minimap, screen, info, msize, ssize, num_action):
	
		mconv1 = tf.keras.layers.Conv2D(		 filters=16,
												 kernel_size=5,
												 strides=(1,1),
                                                 padding='same'
												 )(tf.transpose(minimap, [0, 2, 3, 1]))
		
		mconv2 = tf.keras.layers.Conv2D(
												 filters=32,
												 kernel_size=3,
												 strides=(1,1),
                                                 padding='same'
												 )(mconv1)
		
		sconv1 = tf.keras.layers.Conv2D(
												 filters=16,
												 kernel_size=5,
												 strides=(1,1),
                                                 padding='same'
												 )(tf.transpose(screen, [0, 2, 3, 1]))
		
		sconv2 = tf.keras.layers.Conv2D(
												 filters=32,
												 kernel_size=3,
												 strides=(1,1),
                                                 padding='same'
												 )(sconv1)
		
		info_fc = tf.keras.layers.Dense(units=256,activation=tf.tanh)(layers.flatten(info))

	# Compute spatial actions
		feat_conv = tf.concat([mconv2, sconv2], axis=3)
		
		spatial_action = tf.keras.layers.Conv2D(filters=1,kernel_size=1,strides=(1,1),padding='same')(feat_conv)
		spatial_action = tf.nn.softmax(layers.flatten(spatial_action))
		
	# Compute non spatial actions and value
		feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
		feat_fc = tf.keras.layers.Dense(units=256,activation=tf.nn.relu)(feat_fc)
		non_spatial_action = tf.keras.layers.Dense(units=num_action,activation=tf.nn.softmax)(feat_fc)
		value = tf.reshape(tf.keras.layers.Dense(units=1,activation=None)(feat_fc), [-1])

		return spatial_action, non_spatial_action, value