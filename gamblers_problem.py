"""
Gambler's problem: value iteration
"""
import random
import time

# TODO: results don't match book plots for p = 0.4, gamma = 0.9

# Psuedo-code:
# Do
#   delta = 0
#   For state in states:
#       old_v = Value(state) # (our current estimate)
#       new_v = max of all {probability(s->s')*[reward(s') + gamma * Value(s')]}
#       Value(state) = new_v
#       delta = max(delta, old_v - new_v)
# While delta < theta
#
# Best action : max of all wagers -> wagers is two outcomes weighted by probability p


    
goal = 100
states = range(goal)
values = [0]*(goal+1) # Value estimate
#values[goal] = 1
policy = [0]*goal # Wager @ state
p = 0.4 # Probability of 'heads'
gamma = 0.9 # Weight
theta = 0.01 # Termination accuracy

def max_wager(state):
    return min(state, goal-state)
    
def reward(state):
    if state == goal:
        return 1
    return 0

def weighted_reward(state, wager):
    # Check this!!!
    reward_p = p * ( reward(state + wager) + gamma*values[state + wager] )
    reward_n = (1-p) * ( reward(state - wager) + gamma*values[state - wager] )
    return reward_p + reward_n
    
def best_value(state):
    return max([weighted_reward(state, wager) for wager in range(max_wager(state)+1)])
    
def best_wager(state):
    wagers = [(weighted_reward(state, wager), wager) for wager in range(max_wager(state)+1)]
    return max(wagers)[1]
    
def iter_values(state):
    global values
    old_v = values[state]
    new_v = best_value(state)
    values[state] = new_v
    delta = old_v - new_v
    print delta

# Results are too jerky, should be smoother
def iter_states():
    global values
    for state in states:
        old_v = values[state]
        # Is the value of a state the value of the best wager?
        new_v = best_value(state)
        values[state] = new_v
        delta = old_v - new_v
        print delta

# Value functions creates triangle (betting min(s, goal-s))
def calc_policy(values):
    wagers = [0]*goal
    for i in range(goal):
        wagers[i] = best_wager(i)
    return wagers

# Replicate book plots: