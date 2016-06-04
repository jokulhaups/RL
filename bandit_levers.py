"""
Module for running experiments with the n-bandit problem, based on Sutton ch.2: 
https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node16.html#fig:bandits-graphs
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from functools import partial

class Lever(object):
    '''A random reward with some mean and variation'''
    def __init__(self, mu, sigma):
        self._next = partial(random.gauss, mu, sigma)
        self.count = 0
             
    def pull(self):
        self.count += 1
        return self._next()
        
class LeverModel(object):
    '''Our current expected reward for the corresponding lever'''
    def __init__(self, expected_reward = 0):
        self.expected_reward = 0
        self.counts = 0.

    def value(self):
        return self.expected_reward
        
    def update(self, reward):
        '''update expected reward by online running average'''
        weighted_average = self.expected_reward * self.counts
        self.counts += 1.
        self.expected_reward = (weighted_average + reward) / self.counts

class Bandit(object):
    '''A lever-puller trying to maximize his reward'''
    def __init__ (self, lever_list, model_default = 0):
        self.levers = lever_list
        self.scores = []
        self.plays  = []
        self.models = [LeverModel(model_default) for i in range(len(lever_list))]
    
    def pick_lever(self): 
        raise NotImplemented('pick_lever should be defined in derived class!')
        
    def gamble(self, verbose = False):
        '''Pick lever, pull it, and update models'''
        lever_num = self.pick_lever()
        value = self.levers[lever_num].pull()
        self.scores.append(value); self.plays.append(lever_num)
        self.models[lever_num].update(value)
               

class EpsilonBandit(Bandit):
    '''An epsilon-greedy lever-puller'''
    def __init__ (self, lever_list, epsilon):
        super(EpsilonBandit, self).__init__(lever_list)
        self.epsilon = epsilon
    
    def reset(self):
        self.__init__(self.levers, self.epsilon)
        
    def pick_lever(self):
        '''Pick to balance exploration and exploitation, return lever #'''
        if random.random() < self.epsilon: # choose randomly
            return random.randrange(len(self.levers))
        else:
            return self.greedy_pick()
        
    def greedy_pick(self):
        '''Always pick lever with max expected reward, return lever #'''
        values = map(LeverModel.value, self.models)
        return values.index(max(values))

class SoftmaxBandit(Bandit):
    '''A Boltzmann distribution lever-puller'''
    def __init__ (self, lever_list, tau = 1):
        super(SoftmaxBandit, self).__init__(lever_list, 1.)
        self.tau = tau
        self.rebuild_probs() # vector that sums to 1
        self.expected_value = []
    
    def reset(self):
        self.__init__(self.levers, self.tau)
        
    def softmax(self, x):
        if type(x) is list:
            x = np.array(x)
        return np.exp(x/self.tau)
        
    def rebuild_probs(self):
        '''Update softmax probs'''
        values = self.softmax(map(LeverModel.value, self.models))
        self.probs = values / sum(values)

    # Probabilty of picking lever:  e^(value/t) / sum (e^(value[i])/t)
    def pick_lever(self):
        '''Pick to balance exploration and exploitation, return lever #'''
        self.rebuild_probs()
        # Cheating, because we know the lever scores
        self.expected_value.append(sum(self.probs * range(10)))
        cdf_max = random.random()
        cdf = 0 
        for i, prob in enumerate(self.probs):
            cdf += prob
            if cdf > cdf_max:
                return i
        raise RuntimeError("Softmax failed to pick a value")

class Experiment(object):
    '''Run a series of trials, and hold the metrics/results.  Helps smooth
    the noisy data from single runs and make meaningful comparisons between
    strategies.  TODO: multi-thread trials
    '''
    def __init__(self, bandits, trials = 1000, steps = 2500):
        self.bandits = bandits
        self.trials, self.steps = trials, steps
        # Track 2 metrics per bandit: average reward and % of time we choose
        #   optimally (both vs. time to show learning rate)
        self.bandit_scores = []
        self.bandit_plays = []
    
    def reset_trial(self):
        for bandit in self.bandits:
            bandit.reset()
    
    def run_trial(self):
        for i in range(self.steps):
            map(Bandit.gamble, self.bandits)
    
    def run_experiment(self, best_lever):
        for i in xrange(self.trials):
            if (i+1) % 100 == 0 and i > 0:
                print 'Trial {}'.format(i+1)
            # reset and run trial
            self.reset_trial()       
            self.run_trial()
            # collect results (scores, plays)
            self.bandit_scores.append([b.scores for b in self.bandits])
            self.bandit_plays.append([b.plays for b in self.bandits])
        avg_rewards = np.average(self.bandit_scores, axis = 0)
        pct_optimal = 100.*np.average([[[b==best_lever for b in c] for c in l]\
                               for l in self.bandit_plays], axis = 0)
        print ("Done!")
        return avg_rewards, pct_optimal
        
if __name__ == '__main__':
    # Setup levers and bandits
    levers = [Lever(i*0.25, 1) for i in range(10)] # best lever is 9
    #epsilons = [0, 0.01, 0.1]
    #bandits = [EpsilonBandit(levers, e) for e in epsilons]
    taus = [.2, .5, 1]
    bandits = [SoftmaxBandit(levers, t) for t in taus]
    
    # Setup and run a bunch of trials:
    num_trials = 2000
    E = Experiment(bandits, trials = num_trials, steps = 2000)
    avg_reward, pct_optimal = E.run_experiment(best_lever = 9)

    # Plot results
    plt.figure(1)
    for result in avg_reward:
        plt.plot(result)
        
    plt.legend(map(lambda x: r'$\tau$={}'.format(x), taus))
    plt.title('Average rewards, {} trials'.format(num_trials))
    plt.grid(); plt.ylabel('Average reward'); plt.xlabel('Plays')
    
    plt.figure(2)
    for result in pct_optimal:
        plt.plot(result)
        
    plt.legend(map(lambda x: r'$\tau$={}'.format(x), taus))
    plt.title('Percent optimal play, {} trials'.format(num_trials))
    plt.grid(); plt.ylabel('%'); plt.xlabel('Plays')    