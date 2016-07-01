"""
Module for running experiments with policy iteration, based on Sutton ch. 4: 
https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node43.html
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Iteration is really (argmax) 1-step lookahead average over all probabilities
# Value, Action, Next State

# delta = 0
# For each state do:
#    v = V[state]
#    V[state] = Sum over s' (Prob s->s') * (Reward s->s' + gamma V[s'])
#    delta = max(delta, v-V[s])
# while delta < theta


def poisson(n, l):
    '''Poisson probability of n with expectation l (lambda)'''
    return l**n * np.exp(-l) / math.factorial(n)

def prob_n_or_more(n, l):
    '''Poisson probability of n or more for each n with expectation l'''
    pp = [poisson(i, l) for i in range(n)]
    cdf = np.ones_like(pp)
    cdf[0] = 1 - pp[0]
    for i in range(1, n):
        cdf[i] = cdf[i-1] - pp[i]
    return cdf

def value(a, b, la, lb):
    '''Expected value of state [a, b] with [lambda_a, 'lambda_b]'''
    return sum([poisson(i, la) for i in range(a)]) +\
           sum([poisson(i, lb) for i in range(b)])
    
def state_values(n, l):
    '''Return immediate value of all states 0 to n @ poisson(l)'''
    # N cars * probability
    return [i*poisson(i, l) for i in range(n)]
    
def pvalues(n, l):
    '''Return actual value of states up to n with expectation lambda'''
    pn = prob_n_or_more(n, l) # to test probability of capturing the value
    vals = np.zeros(n)
    for i in range(1, n):
        vals[i] = vals[i-1] + pn[i] # how probable is demand for the extra car?
    return vals

def cost_to_go(steps):
    return abs(steps) * 2

def step_value(x, y, step, matrix):
    if x+step < 0 or y-step < 0:
        raise ValueError("({},{}) step {} is out of range".format(x,y,step))
    return (matrix[x+step][y-step] - matrix[x][y]) - cost_to_go(step)

# Are we ignoring time?  i.e. have to sum policy / state for some t to get value?
def best_policy_at_point(matrix, a, b):
    '''return best policy and new state value for matrix at point a, b'''
    max_a_to_b = min(a, size-b-1, 5)
    max_b_to_a = min(b, size-a-1, 5)
    steps = range(-max_a_to_b, max_b_to_a+1)
    step_values = [step_value(a, b, step, matrix) for step in steps]
    delta = max(step_values)
    best_policy = steps[step_values.index(delta)]
    return best_policy, delta
    
def iterate(matrix):
    '''Create new policy and matrix based on current matrix'''
    new_state = np.zeros_like(matrix)
    policy = np.zeros_like(matrix)
    for r, row in enumerate(matrix):
        for c, val in enumerate(row):
            p, delta = best_policy_at_point(matrix, r, c)
            policy[r][c] = p
            new_state[r][c] = val + delta
            
    return policy, new_state
            
            
# create state matrix by summing rows / columns:
size = 20
cost = 10
scale = size * cost
row = pvalues(size, 3) * cost # all rows are the same!
col = pvalues(size, 4) * cost # all cols are the same!
prob_returns_a = prob_n_or_more(size, 3)
prob_returns_b = prob_n_or_more(size, 2)

# Actual values are greater, due to returns:
for i in range(size):
    row_sum = 0
    col_sum = 0
    for j in range(size-i):
        row_sum += prob_returns_a[j] * row[i+j] # add the value of j returns
        col_sum += prob_returns_b[j] * col[i+j]
        # We now have this many cars!!!
    row[i] += row_sum
    col[i] += col_sum

rows = [row for i in range(size)]
cols = [col for i in range(size)]

# How to account for value / probability of returns???
state_matrix = np.transpose(rows) + cols


plt.figure()

# Simple contour
#CS = plt.contour(np.arange(size), np.arange(size), rows)
#plt.clabel(CS, inline=1, fontsize=10)
#plt.title('Simplest default with labels')

# Contour with color map
im = plt.imshow(state_matrix, interpolation='none', origin='lower', cmap=cm.gray)
CBI = plt.colorbar(im, shrink=0.8)
#levels = np.arange(-1.2, 1.6, 0.2)
# CS = plt.contour(state_matrix, levels,
#                 origin='lower',
#                 linewidths=2,
#                 extent=(-3, 3, -2, 2))
#CS = plt.contour(np.arange(size), np.arange(size), state_matrix)
#CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.ylabel('$n_a')
plt.xlabel('$n_b')

plt.show()

new_p, new_s = iterate(state_matrix)
plt.figure()
im = plt.imshow(new_p, interpolation='none', origin='lower', cmap=cm.gray)
plt.colorbar(im)
new_p2, _ = iterate(new_s)
plt.figure()
im = plt.imshow(new_p2, interpolation='none', origin='lower', cmap=cm.gray)
plt.colorbar(im)
