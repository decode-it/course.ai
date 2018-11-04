Multilayer Perceptron Exercises
===============================

## Input/output mapping

### Single neuron models

Load each of the `data/mlp.*_neuron.naft` models and investigate visually the effect of changing the weight and bias parameters:

* What happens when the weight is increased?
* What happens when the bias is increased?

### Single hidden layer models

#### Linear hidden layer model

Load the `data/mlp.linear_hidden_layer.naft` model and set its parameters manually, or randomly:

* Is there any advantage of using _multi-layered_ linear models over a single linear neuron?
* Is there any advantage of using _multi-layered_ non-linear models over a single non-linear neuron?

#### Tanh hidden layer model

Load the `data/mlp.tanh_hidden_layer.naft` model.

Activate one neuron in the hidden layer by setting the weight from `input` to `hidden tanh neuron` to 1, and from `hidden tanh neuron` to `output linear neuron` to 1:

* What is the effect of changing the bias from `input` to `hidden tanh neuron`?
* What is the effect of changing the bias from `hidden tanh neuron` to `output linear neuron`?

Play around with its parameters to produce a Gaussian-like function. You can do this with only 2 neurons from the hidden layer.

### Deep models

Draw on paper an _deep neural network_ with:

* 1 input
* 4 hidden layers, 3 neurons per layer
* 1 output neuron

* How many parameters does such a model have?
* Validate your computation by loading the `data/mlp.tanh_deep_network.naft` model. The `ai.app` displays the number of weights the model contains.


## Data cost and model complexity

### Parameter norm

Load `data/mlp.tanh_deep_network.naft` and initialize its parameters randomly, making the output is visible. Next, use the weight scaler to increase/decrease all the weight and bias parameters together:

* What is the effect of increasing weights?

### Data cost

Load the `data/data.noisy_linear.naft` dataset and `data/mlp.linear_neuron.naft` model. Try to change the parameters to create the best fit for the data:

* What is the minimal cost you can achieve? Note it down.
* What happens with the `data cost` and `cost` curve when you do this?
* What happens with the `weight cost` when you increase parameters?

Repeat the same exercise for `data/data.noisy_sine.naft` and `data/mlp.tanh_hidden_layer.naft`:

* What is the minimal cost you can achieve? Note it down.


## Learning from data

### Exhaustive learning

Exhaustive learning will change selected model parameters independently, effectively walking a grid. To use the exhaustive learning mode, you have to:

* Load a model
* Load a dataset
* Select `Exhaustive` learning mode. When learning is not yet activated, selecting `Exhaustive` learning mode will add checkboxes next to the model parameters. These checkboxes allow the user to select the parameters that should be learned.
* Set the desired number of points to scan.
* Enable the `Learn` checkbox. Once exhaustive learning search the grid for the optimal parameters, it will switch into `NoLearn` learning mode.

#### Linear regression

Load the `data/data.noisy_linear.naft` dataset and `data/mlp.linear_neuron.naft` model. Use exhaustive learning to search the weight and bias space for this simple model.

* Note down how long this learning mode takes for e.g., 40 steps in 2 dimensions.
* Can it do better than your manual learning above?

#### Linear regression
            [how long will it take to learn the deep model with the exhaustive method]
            [necessity to randomize the weights before learning]
            [alternating SD]
            [wrong error landscape for SCG]
            [visualize weight decay shape]
            [visualize trade-off between cost and WD]
            [plot path for SD]

