# Multilayer Perceptron from Scratch

Here I implement the multilayer perceptron from "scratch" using `numpy`. See ` report.pdf` for details on the neural network and project results. The implementation is derived solely from the readings and sources referenced in the _References_ section of the `report.pdf `. I compare my implementation with a reference `tensorflow `implementation and make use of K-fold cross validation with the construction of confidence intervals to validate my models. The `report.pdf` also contains explanations on the foundational concepts of neural networks for personal elucidation.

# Reproducing Results

The `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `tensorflow `packages are needed for this repository. I recommend using a `conda` environment and running the following commands:

```
conda create --name mlp-from-scratch
conda activate mlp-from-scratch
conda install numpy scipy scikit-learn
conda install -c conda-forge tensorflow matplotlib
```

You can copy and paste the first or second line of `experiments.txt` into the terminal to reproduce my results.

The script `demo.py` was used to run experiments as well as debug the multilayer perceptron. The list of arguments to `demo.py` can be shown by runnning `python demo.py -h`, though I list the arguments below as well. Note that the `random-classification` task only supports binary classification for now.

```
$ python demo.py -h
usage: demo.py [-h] [--debug {True,False}] [--inspect-model-history {True,False}] [--random-seed RANDOM_SEED] [--verbose {True,False}] [--m-examples M_EXAMPLES]
               [--n-features N_FEATURES] [--t-targets T_TARGETS] [--c-categories C_CATEGORIES] [--num-layers NUM_LAYERS] [--num-hidden-units NUM_HIDDEN_UNITS]
               [--test-size TEST_SIZE] [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS] [--learning-rate LEARNING_RATE] [--confidence-level CONFIDENCE_LEVEL]
               [--n-kfold-iterations N_KFOLD_ITERATIONS] [--bar-chart-path BAR_CHART_PATH] [--bar-plot-title BAR_PLOT_TITLE] [--bar-width BAR_WIDTH]
               {breast-cancer-classification,diabetes-regression,random-regression,random-classification}

script for testing MLP from scratch.

positional arguments:
  {breast-cancer-classification,diabetes-regression,random-regression,random-classification}
                        specify supervised learning task.

optional arguments:
  -h, --help            show this help message and exit
  --debug {True,False}  bool to debug
  --inspect-model-history {True,False}
                        bool to print model history at the end of N, KFold CV
  --random-seed RANDOM_SEED
                        random seed for reproducibility. (default: 0)
  --verbose {True,False}
                        whether to print model fitting output. (default: False)

random-data-params:
  parameters for random dataset tasks

  --m-examples M_EXAMPLES
                        number of training examples. (default: 32)
  --n-features N_FEATURES
                        number of features. (default: 1)
  --t-targets T_TARGETS
                        number of targets for regression. (default: 1)

hyperparameters:
  hyperparametesr for MLP.

  --num-layers NUM_LAYERS
                        number of hidden layers. (default: 1)
  --num-hidden-units NUM_HIDDEN_UNITS
                        number of hidden units in hidden layers. (default: 2)
  --test-size TEST_SIZE
                        percent of data to devote to testing. (default: 0.2)
  --batch-size BATCH_SIZE
                        batch size for training. (default: 4)
  --num-epochs NUM_EPOCHS
                        number of epochs to fit model. (default: 2)
  --learning-rate LEARNING_RATE
                        learning rate for gradient descent. (default: 1e-2)

stats:
  parameters for report statistics.

  --confidence-level CONFIDENCE_LEVEL
                        confidence level for intervals (aka alpha). (default: 0.95)
  --n-kfold-iterations N_KFOLD_ITERATIONS
                        number of times to conduct k-fold cross validation. Also changes random state for shuffling n times. (default: 1)

figures:
  parameters for figures.

  --bar-chart-path BAR_CHART_PATH
                        path to save bar chart. (default: ./tex/figures/bar_chart.svg)
  --bar-plot-title BAR_PLOT_TITLE
                        title for bar plot. (default: N, K-Fold CV Model Performance Comparison)
  --bar-width BAR_WIDTH
                        width of bars in bar charts. (default: 0.25)
```
