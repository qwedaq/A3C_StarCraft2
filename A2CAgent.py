from pysc2.agents import base_agent
import tensorflow as tf

class A2CAgent(base_agent):
    """ A2C agent """

    def __init__(self):
        super(A2CAgent,self).__init__()
        buildNetwork()

    def step(self,obs):
        super(A2CAgent,self).step(obs)
        
    def buildNetwork():
        minimap = tf.placeholder(dtype=tf.float32,shape=[None,map_size])