!pip install graphviz

import graphviz

# Create a directed graph for the Neural Network architecture
dot = graphviz.Digraph(comment='Fraud Detection NN Architecture', 
                        graph_attr={'rankdir':'LR', 'splines':'line'})

# Define Input Features
inputs = ['TransactionAmount', 'TransactionTime', 'MerchantCategory', 
          'CustomerAge', 'AccountBalance', 'NumberOfTransactionsToday']

# 1. Add Input Layer
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='Input Layer (Features)', color='blue')
    for i, feature in enumerate(inputs):
        c.node(f'in_{i}', feature, shape='box')

# 2. Add Hidden Layer 1 (8 Neurons for pattern recognition)
with dot.subgraph(name='cluster_1') as c:
    c.attr(label='Hidden Layer 1 (ReLU)', color='green')
    for i in range(8):
        c.node(f'h1_{i}', f'Neuron {i+1}', shape='circle')

# 3. Add Hidden Layer 2 (4 Neurons for abstraction)
with dot.subgraph(name='cluster_2') as c:
    c.attr(label='Hidden Layer 2 (ReLU)', color='green')
    for i in range(4):
        c.node(f'h2_{i}', f'Neuron {i+1}', shape='circle')

# 4. Add Output Layer (Sigmoid activation for probability)
with dot.subgraph(name='cluster_3') as c:
    c.attr(label='Output Layer (Sigmoid)', color='red')
    c.node('out', 'Fraud Probability\n(0 to 1)', shape='doublecircle')

# 5. Add Connections (Edges)
# Connect Inputs to Hidden 1
for i in range(len(inputs)):
    for j in range(8):
        dot.edge(f'in_{i}', f'h1_{j}', alpha='0.2')

# Connect Hidden 1 to Hidden 2
for i in range(8):
    for j in range(4):
        dot.edge(f'h1_{i}', f'h2_{j}')

# Connect Hidden 2 to Output
for i in range(4):
    dot.edge(f'h2_{i}', 'out')

display(dot)
