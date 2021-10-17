import random
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import copy


w0_true = 1.0
w1_true = 1.0
w2_true = 1.0
w3_true = 1.0
error_spread = 10.0

# this is the decision boundary that separates points in class 1
# from points in class 2, for any 2D datapoint p = (x,y), p is in Class 1
# iff y <= f_true(x). Otherwise p is in Class 2.
# For reference:
f_true = lambda X: w3_true * np.power(X, 3) + w2_true * np.power(X, 2) + w1_true * X + w0_true



# Now, let's generate a candidate dataset. This is just a bunch of random datapoints
# that follow this rule.
N = 50

X = np.random.uniform(low=-10, high=10, size=(N,))
Y = f_true(X) + np.random.normal(loc=0.0, scale=error_spread, size=(N,))


# Language for Genetic Optimization

term_set = [
    '0',
    '1',
    'x'
]

function_set = [
    ('+', 2),
    ('*', 2)
]

definitions = {
    '+' : lambda x, y: x + y,
    '*' : lambda x, y: x * y,
    '0' : 0,
    '1' : 1
}

term_fraction = len(term_set) / (len(term_set) + len(function_set))

#
# Population Initialization Code
#

def get_random_element(data):
    return data[random.randint(0, len(data) - 1)]


def gen_random_expr(f_set, t_set, max_d, method):
    if max_d == 0 or (method == "grow" and np.random.uniform() < term_fraction):
        expr = get_random_element(term_set)
    else:
        (symbol, arity) = get_random_element(function_set)
        args = []

        for i in range(arity):
            args.append(gen_random_expr(f_set, t_set, max_d - 1, method))

        expr = [symbol, *args]

    return expr


#
# Selection Procedures
#

def partition_population(programs, n=2):
    div = len(programs) / float(n)
    return [ programs[int(round(div * i)):int(round(div * (i+1)))] for i in range(n)]


def eval_programs(X, Y, programs):
    program_y_preds = [list(eval_expr(p, {'x': x}) for x in X) for p in programs]
    program_costs = []

    for y_pred in program_y_preds:
        cost = np.power(Y - np.array(y_pred), 2).mean() # MSE Cost
        program_costs.append(cost)

    return np.array(program_costs)



def tournament_selection(X, Y, programs):
    program_costs = eval_programs(X, Y, programs)
    best = np.argmax(program_costs)

    return programs[best], program_costs[best]


def subtree_crossover(parent_A, parent_B):
    pass

def choose_random_subtree(tree):
    pass

def subtree_mutation(parent_A, parent_B):
    pass

#
# Evaluate an evolved progrogram
#

def eval_expr(expr, context):
    if type(expr) == str:
        if expr in context:
            return context[expr]
        else:
            return definitions[expr]

    elif type(expr) == list:
        func = definitions[expr[0]]
        args = expr[1:]
        evaluated_args = map(lambda e: eval_expr(e, context), args)

        return func(*evaluated_args)

# For an node in the tree, it returns the [p(left), p(node), p(right)]
#
def get_probabilities(expr):
    if type(expr) == str:
        return [0.1]

    if type(expr) == list:
        func = definitions[expr[0]]
        args = expr[1:]
        evaluated_args = map(lambda e: get_probabilities(e), args)
        flat_args = reduce(lambda a,b: a + b, evaluated_args)
        return [0.9] + flat_args


def get_expr_CDF(expr):
    probabilities = get_probabilities(expr)
    probs_sum = sum(probabilities)
    CDF = list(map(lambda x: x / probs_sum, probabilities))
    CDF = reduce(lambda a, b: (b + a[0], a[1] + [a[0]]), CDF, (0, []) )[1]

    return CDF

def get_weighted_random_index(CDF):
    r = np.random.uniform()
    return np.searchsorted(CDF, r, side='left') - 1


def get_subtree_at_index(expr, expr_index):
    if expr_index == 0:
        return expr, -1

    elif type(expr) == str:
        return expr, expr_index

    elif type(expr) == list:
        args, arg_index = expr[1:], 0
        while arg_index < len(args) and expr_index > -1:
            subexpr, expr_index = get_subtree_at_index(args[arg_index], expr_index - 1)
            arg_index += 1

        return subexpr, expr_index


def replace_subtree_at_index(expr, expr_index, subtree):
    if expr_index == 0:
        return subtree, -1

    elif type(expr) == str:
        return expr, expr_index

    elif type(expr) == list:
        args, arg_index = expr[1:], 0
        while arg_index < len(args) and expr_index > -1:
            subexpr, expr_index = replace_subtree_at_index(args[arg_index], expr_index - 1, subtree)
            expr[arg_index + 1] = subexpr
            arg_index += 1

        return expr, expr_index





fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

def plot(X, Y, programs):
    ax1.cla()
    ax1.set_title('Data / True Function')
    ax1.scatter(X, Y, color="blue")

    input = np.linspace(-10, 10, 50)
    output_true = f_true(input)

    ax1.plot(input, output_true, color="lightblue")

    program_costs = eval_programs(X, Y, programs)
    program_outputs = [list(eval_expr(p, {'x': x}) for x in input) for p in programs]
    best_programs = np.argsort(program_costs, axis=0)

    for i, ax in enumerate([ax2, ax3, ax4]):
        ind = best_programs[i]
        print(programs[ind])

        ax.cla()
        ax.set_title('program %i' % ind )
        ax.plot(input, output_true, color="lightblue")
        ax.plot(input, program_outputs[ind], color='red')


    plt.pause(0.25)


D = 3 # program tree depth
N = 10 # initial samples
K = 5 # number of programs to mutate.

programs_grow = list(map(lambda d: gen_random_expr(function_set, term_set, d, "grow"), [D] * (N // 2)))
programs_full = list(map(lambda d: gen_random_expr(function_set, term_set, d, "full"), [D] * (N // 2)))
programs = programs_grow + programs_full

p = ['*', ['+' , '1', '1'], '0']
st = ['*', '1', ['+', 'x', '1']]

# plot(X, Y, programs)

# plt.show()

# 1000 generations
for i in range(1000):

    # random.shuffle(programs)
    # [population_A, population_B] = partition_population(programs, n=2)
    # print(programs)

    population_A = random.sample(programs, 5)
    population_B = random.sample(programs, 5)

    parent_A, parent_A_cost = tournament_selection(X, Y, population_A)
    parent_B, parent_A_cost = tournament_selection(X, Y, population_B)


    # Select a subtree from parent A
    parent_A_CDF = get_expr_CDF(parent_A)
    parent_A_i = get_weighted_random_index(parent_A_CDF)
    parent_A_st, _ = get_subtree_at_index(parent_A, parent_A_i)


    parent_B_CDF = get_expr_CDF(parent_B)
    parent_B_i = get_weighted_random_index(parent_B_CDF)

    child = copy.deepcopy(parent_B)

    child, _ = replace_subtree_at_index(child, parent_B_i, parent_A_st)

    # print('Parent A (Donor) [%s]:' % parent_A_i)
    # print(parent_A)
    # print('Parent B (Recipient) [%s]:' % parent_B_i)
    # print(parent_B)
    # print('Child:')
    # print(child)

    programs.append(child)

    plot(X, Y, programs)

    # exit()

plt.show()
