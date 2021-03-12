# ==============================================================================

"""
Description: 
    This is the Tensorflow implementation for the INDIAN algorithm based on the paper
    "an Inertial Newton Algorithm for Deep Learning" by C. Castera, J. Bolte, 
    C. Fevotte and E. Pauwels (https://arxiv.org/abs/1905.12278)
    
    This algorithm is inspired by Dynamical systems and Newton's Third Law and the study
    of dynamical systems. Its two hyperparameters alpha and beta have a meaningful 
    interpretation in mechanics which helps the tuning as described in the original paper.

Args:
    lr: The initial learning rate, can be a a float or a tensor of floats. We strongly recommend to tune
      this parameter (as for any other optimizer).
    alpha: The first and most important hyperparameter, the 'viscous damping coefficient'. Default value
      is 0.5 but can be set to any positive value. Usually high value cause slow but stable training while 
      low values of alpha can speed things up but might crash. This parameter can also be tuned to
      reach a good tradeoff between validation and training (See original paper for more indications)
    beta: The second hyperparameter, 'The Newtonian effect coefficient'. Default is 0.1 but it can be 
      useful to try 1 other powers of ten (say 0.01, 1.0 and 10 for instance).This parameter can also 
      be tuned to reach a good tradeoff between validation and training (See original paper for more indications)
    decay: the learning rate is decreased at iteration k with the formula:
      lr_k = lr * decay /(k+1)**(decaypower) . Default is 1.0, we recommend to let this value unchanged
    decaypower: see previous parameter for the explanation. Default is 1/2. Values such as 1/4, or 1/8 can
    speed the traning phase (but reduce robustness to noise).
    speed_ini: The initial velocity is in the direction of the speepest descent:
    it is in exactly - speed_ini* gradient. Default is 1., we recommend to keep this parameter unchanged.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.train import get_global_step as num_iter
from tensorflow import cond
from tensorflow.math import equal

@tf_export(v1=["train.AdamOptimizer"])
class INNAOptimizer(optimizer.Optimizer):
  """Optimizer that implements the INNA algorithm.
  See [Castera et al., 2019](https://arxiv.org/abs/1905.12278).
  """

  def __init__(self,
                 lr=0.01,
                 alpha=0.5,
                 beta=0.1,
                 decay=1.,
                 decaypower = 0.5,
                 speed_ini=1.0,
                 epsilon=1e-8,
                 use_locking=False,
                 name="INNA"):
   
    super(INNAOptimizer, self).__init__(use_locking,name)
    self._iterations = 0
    self._lr = lr
    self._alpha = alpha
    self._beta = beta
    self._epsilon = epsilon
    self._decay = decay
    self._decaypower = decaypower
    self._speed_ini = speed_ini

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._alpha_t = None
    self._beta_t = None
    self._epsilon_t = None
    self._decay_t = None
    self._decaypower_t = None
    self._speed_ini_t = None


  def _create_slots(self, var_list):
    # Create slots for the auxiliary variable.
    for v in var_list:
      self._zeros_slot(v, "v1", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    alpha = self._call_if_callable(self._alpha)
    beta = self._call_if_callable(self._beta)
    epsilon = self._call_if_callable(self._epsilon)
    decay = self._call_if_callable(self._decay)
    decaypower = self._call_if_callable(self._decaypower)
    speed_ini = self._call_if_callable(self._speed_ini)
    

    self._lr_t = ops.convert_to_tensor(self._lr, name="lr")
    self._alpha_t = ops.convert_to_tensor(self._alpha, name="alpha")
    self._beta_t = ops.convert_to_tensor(self._beta, name="beta")
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
    self._decay_t = ops.convert_to_tensor(self._decay, name="decay")
    self._decaypower_t = ops.convert_to_tensor(self._decaypower, name="decaypower")
    self._speed_ini_t = ops.convert_to_tensor(self._speed_ini, name="speed_ini")

  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
    beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    decay_t = math_ops.cast(self._decay_t, var.dtype.base_dtype)
    decaypower_t = math_ops.cast(self._decaypower_t, var.dtype.base_dtype)
    speed_ini_t = math_ops.cast(self._speed_ini_t, var.dtype.base_dtype)
    

    v = self.get_slot(var, "v1")
    #(1.-self.alpha*self.beta)*p )
    #Initialise v such that the initial speed is in the direction of -grad
    v_temp = cond( equal(num_iter(),0) ,
      lambda : (1.-alpha_t*beta_t) * var - beta_t**2 * grad + beta_t * speed_ini_t * grad, lambda : v )

    v_t = v.assign( v_temp - ( lr_t * decay_t / math_ops.pow(math_ops.cast(num_iter()+1, var.dtype.base_dtype),decaypower_t) ) * ( (alpha_t-1./beta_t) * var + 1./beta_t * v_temp ) )
   
    var_update = state_ops.assign_sub( var, ( lr_t * decay_t / math_ops.pow(math_ops.cast(num_iter()+1, var.dtype.base_dtype),decaypower_t) ) * ( (alpha_t-1./beta_t) * var + 1./beta_t * v_temp + beta_t * grad ) ) #Update 'ref' by subtracting 'value
    
    return control_flow_ops.group(*[var_update, v_t])


  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported yet.")




    #### EXAMPLE https://github.com/angetato/Optimizers-for-Tensorflow/blob/master/tf_utils/AAdam.py  #####