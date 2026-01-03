from .types import PPOConfig

class EntropyCoefficient(object):

    def __init__(self,
                 cfg: PPOConfig):

        self.fixed_entropy_coef = cfg.fixed_entropy_coef
        self.anneal_entropy_flag = cfg.anneal_entropy_flag
        self.start_entropy_coef = cfg.start_entropy_coef
        self.end_entropy_coef = cfg.end_entropy_coef

    def reset(self,
              total_updates: int):

        if self.anneal_entropy_flag:

            self.slope =\
                (self.end_entropy_coef - self.start_entropy_coef) /\
                max(total_updates - 1, 1)
            
            self.bias = self.start_entropy_coef
        else:
            self.slope = 0
            self.bias = self.fixed_entropy_coef

    def __call__(self,
                 update_idx):

        return self.bias + update_idx * self.slope