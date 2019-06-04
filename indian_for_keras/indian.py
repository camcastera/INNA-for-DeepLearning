from keras.optimizers import *
from keras.legacy import interfaces


class Indian(Optimizer):
    """
    Description: 
        This is the keras implementation for the INDIAN algorithm based on the paper
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
    
    def __init__(self,
                 lr=0.01,
                 alpha=0.5,
                 beta=0.1,
                 decay=1.,
                 decaypower = 0.5,
                 speed_ini=1.0,
                 epsilon=None,
                 **kwargs):
                 
        super(Indian, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.alpha = K.variable(alpha, name='alpha')
            self.beta = K.variable(beta, name='beta')
            self.decay =K.variable(decay, name='decay')
            self.decaypower = K.variable(decaypower,name='decaypower')
            self.initial_decay = decay
            self.speed_ini = speed_ini
            
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
    
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0 :
            lr = lr * (1. / K.pow(1. + self.decay * K.cast(self.iterations,
                                    K.dtype(self.decay)),self.decaypower) )
        
        #psi such that initial speed is orthogonal
        psi = [ K.variable( (1.-self.alpha*self.beta)*p ) for p in params ]
        self.weights =  [self.iterations] + psi
        
        for p, g, v in zip(params, grads, psi) :
            #Warning, (p,v) correspond to (theta,psi) in the paper
            lr_t = lr
            
            #This changes the initial speed (at iteration 1 only)
            v_temp = K.switch( K.equal( self.iterations , 1 ),
                        v - self.beta**2*g + self.beta*self.speed_ini*g , v )
            #
            v_t =  v_temp + lr_t * ( (1./self.beta - self.alpha) * p - 1./self.beta * v_temp  )
            p_t = p + lr_t * ( (1./self.beta - self.alpha) * p - 1./self.beta * v_temp - self.beta * g )
            
            new_p = p_t
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
                
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'alpha': float(K.get_value(self.alpha)),
                  'beta': float(K.get_value(self.beta)),
                  'decay': float(K.get_value(self.decay)),
                  'speed_ini' : self.speed_ini
                 }
        base_config = super(Indian, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
