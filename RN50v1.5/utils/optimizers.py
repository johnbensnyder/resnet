#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

import re
import tensorflow as tf

__all__ = ['FixedLossScalerOptimizer']


class FixedLossScalerOptimizer(tf.train.Optimizer):
    """An optimizer that scales loss and un-scales gradients for FP16 training."""

    def __init__(self, optimizer, scale=None, name="LossScalingOptimizer", use_locking=False):

        super(FixedLossScalerOptimizer, self).__init__(name=name, use_locking=use_locking)

        self._optimizer = optimizer
        self._scale = float(scale) if scale is not None else 1.0

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):

        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)

        gradvar = self._optimizer.compute_gradients(loss, var_list, *args, **kwargs)
        gradvar = [(tf.scalar_mul(1. / self._scale, g), v) for g, v in gradvar]

        return gradvar

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

class MixedPrecisionOptimizer(tf.train.Optimizer):
    """An optimizer that updates trainable variables in fp32."""

    def __init__(self, optimizer,
                 scale=None,
                 name="MixedPrecisionOptimizer",
                 use_locking=False):
        super(MixedPrecisionOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._scale = float(scale) if scale is not None else 1.0

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        if var_list is None:
            var_list = (
                    tf.trainable_variables() +
                    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        replaced_list = var_list

        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)

        gradvar = self._optimizer.compute_gradients(loss, replaced_list, *args, **kwargs)

        final_gradvar = []
        for orig_var, (grad, var) in zip(var_list, gradvar):
            if var is not orig_var:
                grad = tf.cast(grad, orig_var.dtype)
            if self._scale != 1.0:
                grad = tf.scalar_mul(1. / self._scale, grad)
            final_gradvar.append((grad, orig_var))

        return final_gradvar

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

class LarcOptimizer(tf.train.Optimizer):
    """ LARC implementation
        -------------------
        Parameters:
          - optimizer:     initial optimizer that you wanna apply
                           example: tf.train.MomentumOptimizer
          - learning_rate: initial learning_rate from initial optimizer
          - clip:          if True apply LARC otherwise LARS
          - epsilon:       default value is weights or grads are 0.
          - name
          - use_locking
    """

    def __init__(self, optimizer, learning_rate, eta=0.13, clip=True, epsilon=1.,
                 name="LarcOptimizer", use_locking=False):
        super(LarcOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._eta = float(eta)
        self._clip = clip
        self._epsilon = float(epsilon)

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, gradvars, *args, **kwargs):
        v_list = [tf.norm(tensor=v, ord=2) for _, v in gradvars]
        g_list = [tf.norm(tensor=g, ord=2) if g is not None else 0.0
                  for g, _ in gradvars]
        v_norms = tf.stack(v_list)
        g_norms = tf.stack(g_list)
        zeds = tf.zeros_like(v_norms)
        # assign epsilon if weights or grads = 0, to avoid division by zero
        # also prevent biases to get stuck at initialization (0.)
        cond = tf.logical_and(
            tf.not_equal(v_norms, zeds),
            tf.not_equal(g_norms, zeds))
        true_vals = tf.scalar_mul(self._eta, tf.div(v_norms, g_norms))
        # true_vals = tf.scalar_mul(tf.cast(self._eta, tf.float32), tf.div(tf.cast(v_norms, tf.float32), tf.cast(g_norms, tf.float32)))
        false_vals = tf.fill(tf.shape(v_norms), self._epsilon)
        larc_local_lr = tf.where(cond, true_vals, false_vals)
        if self._clip:
            ones = tf.ones_like(v_norms)
            lr = tf.fill(tf.shape(v_norms), self._learning_rate)
            # We need gradients to compute local learning rate,
            # so compute_gradients from initial optimizer have to called
            # for which learning rate is already fixed
            # We then have to scale the gradients instead of the learning rate.
            larc_local_lr = tf.minimum(tf.div(larc_local_lr, lr), ones)
        gradvars = [(tf.multiply(larc_local_lr[i], g), v)
                    if g is not None else (None, v)
                    for i, (g, v) in enumerate(gradvars)]
        return self._optimizer.apply_gradients(gradvars, *args, **kwargs)


class LAMBOptimizer(tf.train.Optimizer):
    """
    LAMBOptimizer optimizer.
    https://github.com/ymcui/LAMB_Optimizer_TF
    # IMPORTANT NOTE
    - This is NOT an official implementation.
    - LAMB optimizer is changed from arXiv v1 ~ v3.
    - We implement v3 version (which is the latest version on June, 2019.).
    - Our implementation is based on `AdamWeightDecayOptimizer` in BERT (provided by Google).

    # References
    - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
    - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
    # Parameters
    - There is nothing special, just the same as `AdamWeightDecayOptimizer`.
    """

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="LAMBOptimizer"):
        """Constructs a LAMBOptimizer."""
        super(LAMBOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/lamb_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/lamb_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

            # Note: Here are two choices for scaling function \phi(z)
            # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
            # identity: \phi(z) = z
            # The authors does not mention what is \gamma_l and \gamma_u
            # UPDATE: after asking authors, they provide me the code below.
            # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
            #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

            r1 = tf.sqrt(tf.reduce_sum(tf.square(param)))
            r2 = tf.sqrt(tf.reduce_sum(tf.square(update)))

            r = tf.where(tf.greater(r1, 0.0),
                         tf.where(tf.greater(r2, 0.0),
                                  r1 / r2,
                                  1.0),
                         1.0)

            eta = self.learning_rate * r

            update_with_lr = eta * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
