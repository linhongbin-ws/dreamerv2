import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from . import dists
from . import tfutils
class SymAgent:

    def __init__(self, agent, act_space, logprob=False, env=None):
        self._agent = agent
        self.act_space = act_space['action']
        self.logprob = logprob
        self.env = env

    def __call__(self, obs, state=None, mode="train"):
        # print("call get oracle")
        if self.env.IsSymEnv:
            _action = self.env.get_sym_action()
            if hasattr(self.act_space, "n"):
                _a = np.zeros(self.act_space.n)
                _a[_action] = 1
                _dist = dists.OneHotDist(probs=tf.convert_to_tensor(_a, np.float32))
            else:
                dist = tfd.Normal(tf.convert_to_tensor(_action, np.float32), 0)
                _dist = tfd.Independent(dist, 1)
            action = _dist.sample(len(obs["is_first"]))
            output = {"action": action}
            if self.logprob:
                output["logprob"] = _dist.log_prob(action)
            return output, None
        else:
            return self._agent(obs, state, mode)

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        return getattr(self._agent, name)
