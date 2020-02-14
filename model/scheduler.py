import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class PiecewiseDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, break_points, learning_rates):
        super(PiecewiseDecay, self).__init__()
        assert len(learning_rates)==len(break_points) + 1
        self.break_points = break_points
        self.learning_rates = learning_rates
        self.current_learning_rate = 0
        self.next_break_point = 0
    
    def __call__(self, step):
        if self.next_break_point<len(self.break_points) and \
            step==self.break_points[self.next_break_point]:
            self.next_break_point+=1
            self.current_learning_rate+=1
        return self.learning_rates[self.current_learning_rate]
    
    def get_config(self):
        return {'break_points': self.break_points,
                'learning_rates': self.learning_rates}
            


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
                "steps_at_scale": self.steps_at_scale, "total_steps": self.total_steps, 
                "decay": self.decay_steps, "name": self.name}