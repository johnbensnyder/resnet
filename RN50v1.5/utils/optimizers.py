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
