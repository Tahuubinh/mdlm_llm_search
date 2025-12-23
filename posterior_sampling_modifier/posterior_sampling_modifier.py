from posterior_sampling_modifier.standard import GumbelSampler
from posterior_sampling_modifier.mcts import MonteCarloTreeSearchSampler

# Available sampling method classes
SAMPLING_METHOD_CLASSES = {
    'standard': GumbelSampler,
    'mcts': MonteCarloTreeSearchSampler,
}
