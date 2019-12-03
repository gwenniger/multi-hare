import torch.optim as optim
from modules.gradient_clipping import GradientClipping

__author__ = "Dublin City University"
__copyright__ = "Copyright 2019, Dublin City University"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Dublin City University Software License (enclosed)"


class MultipleOptimizer(object):
    def __init__(self, op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class Optim(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_at (int, optional): epoch to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, option): custom decay options
    """
    # We use the default parameters for Adam that are suggested by
    # the original paper https://arxiv.org/pdf/1412.6980.pdf
    # These values are also used by other established implementations,
    # e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    # https://keras.io/optimizers/
    # Recently there are slightly different values used in the paper
    # "Attention is all you need"
    # https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    # was used there however, beta2=0.999 is still arguably the more
    # established value, so we use that here as well
    def __init__(self, method, lr, max_grad_norm,
                 lr_decay=1, start_decay_at=None,
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None):
        self.last_ppl = None
        self.lr = lr
        self.original_lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        print("Optim.init. lr " + str(self.lr))
        print("Optim.init. self.lr_decay: " + str(self.lr_decay))
        print("Optim.init. self.statrt_decay_at: " + str(self.start_decay_at))
        print("Optim.init. self.betas[0]: " + str(self.betas[0]))
        print("Optim.init. self.betas[1]: " + str(self.betas[1]))

    def set_parameters_only(self, params):
        self.params = [p for p in params if p.requires_grad]

    def set_parameters(self, params):
        self.params = []
        self.sparse_params = []

        print("params: " + str(params))

        for k, p in params:
            if p.requires_grad:
                if self.method != 'sparseadam' or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.optimizer.state[p]['sum'] = self.optimizer\
                        .state[p]['sum'].fill_(self.adagrad_accum)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,                                betas=self.betas, eps=1e-9)
        elif self.method == 'sparseadam':
            self.optimizer = MultipleOptimizer(
                [optim.Adam(self.params, lr=self.lr,
                            betas=self.betas, eps=1e-8),
                 optim.SparseAdam(self.sparse_params, lr=self.lr,
                                  betas=self.betas, eps=1e-8)])
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def _set_rate(self, lr):
        self.lr = lr
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.lr
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]['lr'] = self.lr

    def step_with_specified_max_norm(self, max_grad_norm):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        if self.max_grad_norm:
            # # First clip by gradient value, in case some gradient values became infinity
            # # this will set them back, which norm-based correction cannot. This
            # # is a somewhat dirty trick to assure that at least the gradient norm can
            # # be correctly computed and does not become nan (because of some infinite
            # # gradient component), which leads to correction not being at all possible
            # GradientClipping.clip_gradient_value(self.params)
            # Then perform the norm-based correction
            made_gradient_norm_based_correction, total_norm = GradientClipping.\
                clip_gradient_norm(self.params, max_grad_norm)
        else:
            print("WARNING: Not Clipping Gradient!")
        self.optimizer.step()

        if self.max_grad_norm:
            return made_gradient_norm_based_correction, total_norm

    def step(self):
        return self.step_with_specified_max_norm(self.max_grad_norm)

    def step_with_scaling_for_size_current_batch(self,
                                                 current_batch_size: int,
                                                 maximum_batch_size: int):
        """
        This method makes a step using norm-based gradient clipping, but correcting for the
        size of the current batch. This is motivated as follows. Suppose the normal batch size
        is 30, and all batches have a gradient norm larger than the max_grad_norm, so the
        gradient is rescaled for each. Now however, for the last batch, which contains the
        (less than 30) remaining examples, the gradient will still be rescaled to the same
        maximum. But since in this batch there are less than 30, say only 2 examples,
        the effective step size per example can become much larger!
        This leads to an over-correction for the last remaining examples in the batch.
        This method compensates for this, by rescaling the norm by  the factor
        current_batch_size / maximum_batch_size. This factor is 1 except for the last batch,
        which contains the "rest" examples. The scaling factor corrects the effective learning
        rate for these "rest" examples, so that the optimizer makes no over-correction for them.

        :param maximum_batch_size:
        :param current_batch_size:
        :return:
        """

        scaled_max_grad_norm = self.max_grad_norm * (float(current_batch_size) / maximum_batch_size)

        return self.step_with_specified_max_norm(scaled_max_grad_norm)

    def update_learning_rate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        
        # Gideon: We don't want to automatically start decaying when the perplexity increases, this should not be a standard behavior
        # eventually this behavior should probably be allowed to be switched on/off in the configuration, but for now the default should be off
        # since the normal standard is Adam optimizer without any feature decay
        #if self.last_ppl is not None and ppl > self.last_ppl:
            #    self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.lr

    def zero_grad(self):
        self.optimizer.zero_grad()

