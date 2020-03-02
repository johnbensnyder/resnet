import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class PiecewiseConstantDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a piecewise constant decay schedule."""

    def __init__(self, initial_learning_rate, scaled_rate, steps_at_scale, boundaries, values, name=None):
        super().__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
                "The length of boundaries should be 1 less than the length of values"
            )
        self.initial_learning_rate = initial_learning_rate
        self.scaled_rate = scaled_rate
        self.steps_at_scale = steps_at_scale
        self.boundaries = boundaries
        self.values = values
        self.name = name
        self.index = tf.Variable(0)

    @tf.function
    def __call__(self, step, dtype=tf.float32):
        step = tf.cast(step, dtype)
        if step<self.steps_at_scale:
            return tf.cast(self.compute_warmup(step), dtype)
        else:
            return tf.cast(self.compute_piecewise(step), dtype)
    
    def compute_warmup(self, step):
        return ((self.scaled_rate*step)+(self.initial_learning_rate*(self.steps_at_scale-step)))/self.steps_at_scale
    
    def compute_piecewise(self, step):
        self.index.assign(0)
        for i, b in enumerate(self.boundaries):
            if step >= b:
                self.index.assign(i+1)
        return tf.convert_to_tensor(self.values)[self.index]
    
    def get_config(self):
        return {"boundaries": self.boundaries, "values": self.values, "name": self.name}
            


class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler that linearly scales up during the first epoch
    then decays exponentially
    """
    def __init__(self, initial_rate, scaled_rate, steps_at_scale, decay_steps, decay_rate, name=None):
        super(WarmupExponentialDecay, self).__init__()
        self.initial_rate = initial_rate
        self.scaled_rate = scaled_rate
        self.steps_at_scale = steps_at_scale
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.name = name
    
    @tf.function
    def __call__(self, step, dtype=tf.float32):
        initial_learning_rate = math_ops.cast(self.initial_rate, dtype)
        decay_steps = math_ops.cast(self.decay_steps, dtype)
        decay_rate = math_ops.cast(self.decay_rate, dtype)
        steps_at_scale = math_ops.cast(self.steps_at_scale, dtype)
        scaled_rate = math_ops.cast(self.scaled_rate, dtype)
        global_step_recomp = math_ops.cast(step, dtype)
        
        if step<=self.steps_at_scale:
            return self.compute_warmup(global_step_recomp, initial_learning_rate, scaled_rate, steps_at_scale)
        else:
            return self.compute_decay(global_step_recomp, scaled_rate, steps_at_scale, decay_rate, decay_steps)
    
    def compute_warmup(self, step, initial_learning_rate, scaled_rate, steps_at_scale):
        return ((scaled_rate*step)+(initial_learning_rate*(steps_at_scale-step)))/steps_at_scale
    
    def compute_decay(self, step, scaled_rate, steps_at_scale, decay_rate, decay_steps):
        return scaled_rate*decay_rate**((step-steps_at_scale)/decay_steps)
    
    def get_config(self):
        return {"initial_rate": self.initial_rate, "scaled_rate": self.scaled_rate, 
                "steps_at_scale": self.steps_at_scale, "decay_rate": self.decay_rate, 
                "decay": self.decay_steps, "name": self.name}