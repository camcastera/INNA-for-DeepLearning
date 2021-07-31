import torch
from torch.optim.optimizer import Optimizer, required


class INNA(Optimizer):
    

    def __init__(self, params, lr=0.1,alpha=0.5,beta=0.1,
                 decaypower=0.,weight_decay=0.):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
            
        if decaypower>0:
            print('Warning: Do not combine the decaypower parameter with a pytorch scheduler')

        defaults = dict(lr=lr, alpha=alpha, beta=beta,
                        decaypower=decaypower,weight_decay=weight_decay)
        super(INNA, self).__init__(params, defaults)
           
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                #state['psi'] = (1.-alpha*beta) * torch.clone(group['params']).detach() 
    '''
    def __setstate__(self, state):
        super(INNA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('psi',(1.-group['alpha']*group['beta'])*torch.clone(group['params']).detach())
    '''     

            
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            lr = group['lr']
            dc = group['decaypower']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                
                #Get the gradient
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # Initialize or get the phase variable
                param_state = self.state[p]
                if 'psi' not in param_state:
                    phase = param_state['psi'] = (1.-alpha*beta) * torch.clone(p.data).detach()
                else:
                    phase = param_state['psi']
                    
                #Prepare the updates    
                phase_update = (alpha-1./beta)*p.data + 1./beta * phase
                geom_update = beta*d_p
                
                #Compute new learning rate
                if dc > 0:
                    lr_t = lr / ( (1 + param_state['step'])** dc )
                else:
                    lr_t = lr
                    
                #Update param and phase
                param_state['psi'].sub_( lr_t , phase_update )
                if weight_decay <= 0:
                    p.data.sub_( lr_t ,  phase_update + geom_update )
                else:
                    WD = weight_decay * p.data
                    p.data.sub_( lr_t ,  phase_update + geom_update + WD)
                param_state['step'] += 1
                
        return loss