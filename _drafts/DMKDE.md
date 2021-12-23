
<a href="https://colab.research.google.com/github/fagonzalezo/qmc/blob/master/examples/qmkde.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Let´s understand **dmkde**

## Install `qmc` 

Let´s install (git clone to the file system of the Colab VM) the module `qmc` which contains
> Custom models inherited from the super class `tf.keras.Model`.

> Custom layers inherited from the super class `tf.keras.layers.Layer`.

More information on customization when using `tf` this can be found at [here](https://www.tensorflow.org/tutorials/customization/custom_layers#models_composing_layers).


```python
# Install qmc if running in Google Colab

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    !rm -R qmc qmc1
    !git clone https://github.com/fagonzalezo/qmc.git
    !mv qmc qmc1
    !mv qmc1/qmc .
else:
    import sys
    sys.path.insert(0, "../")
```

    rm: cannot remove 'qmc': No such file or directory
    rm: cannot remove 'qmc1': No such file or directory
    Cloning into 'qmc'...
    remote: Enumerating objects: 91, done.[K
    remote: Counting objects: 100% (91/91), done.[K
    remote: Compressing objects: 100% (71/71), done.[K
    remote: Total 306 (delta 26), reused 72 (delta 17), pack-reused 215[K
    Receiving objects: 100% (306/306), 20.92 MiB | 24.17 MiB/s, done.
    Resolving deltas: 100% (117/117), done.


## Define visualization tools

First we use the iPython functionality to show richer outputs using the magic command `%matplotlib inline` and force visualizations to be printed.

Then we import two important libraries:
- `numpy` : Support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- `pylab` : Matplotlib is the toolkit, PyPlot is an interactive way to use Matplotlib and PyLab is the same thing as PyPlot but with some extra shortcuts. Using PyLab is discouraged now.

Now let´s create three functions for ploting purposes:

- `plot_data(X,y)`: It has two args, one is the set of points in $\mathbb{R}^2$ and the second one is a vector which contains the class of every point given. It returns a 2D scatter plot with points colored in a rainbow fashion according to their class.
- `plot_decision_region(X,pred_fun)`: It has two args, one is the vector which will define the 2D square plot region(min_x,max_x,vector of points to evaluate the prediction function,...) and the other argument is the prediction function which we are interested to plot. The function returns a 2D contour plot and the points colored with the real class which they belong to.
- `gen_pred_fun`: It has one arg which is a classifier which has a `.predict()` method and from this this functions extracts the predicted values.


```python
%matplotlib inline 
import numpy as np
import pylab as pl

# Function to visualize a 2D dataset
def plot_data(X, y):
    # Subsets y with unique elements in a new array
    y_unique = np.unique(y)
    # Generates a number of colors in rainbow according to the y_unique.size
    colors = pl.cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))    
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        pl.scatter(this_X[:, 0], this_X[:, 1],  c=color,
                    alpha=0.5, edgecolor='k',
                    label="Class %s" % this_y)
    pl.legend(loc="best")
    pl.title("Data")
    
# Function to visualize the decission surface of a classifier
def plot_decision_region(X, pred_fun):
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    min_x = min_x - (max_x - min_x) * 0.05
    max_x = max_x + (max_x - min_x) * 0.05
    min_y = min_y - (max_y - min_y) * 0.05
    max_y = max_y + (max_y - min_y) * 0.05
    x_vals = np.linspace(min_x, max_x, 50)
    y_vals = np.linspace(min_y, max_y, 50)
    XX, YY = np.meshgrid(x_vals, y_vals)
    grid_r, grid_c = XX.shape
    vals = [[XX[i, j], YY[i, j]] for i in range(grid_r) for j in range(grid_c)]
    preds = pred_fun(np.array(vals))
    ZZ = np.reshape(preds, (grid_r, grid_c))
    print(np.min(preds), np.min(ZZ))
    pl.contourf(XX, YY, ZZ, 100, cmap = pl.cm.coolwarm, vmin= 0, vmax=1)
    pl.colorbar()
    CS = pl.contour(XX, YY, ZZ, 100, levels = [0.1*i for i in range(1,10)])
    pl.clabel(CS, inline=1, fontsize=10)
    pl.xlabel("x")
    pl.ylabel("y")

def gen_pred_fun(clf):
    def pred_fun(X):
        return clf.predict(X)[:, 1]
    return pred_fun
```

### Examples 

Example of using `plot_data`


```python
plot_data(np.array([[1,1],[2,2],[3,3]]),np.array([0,1,2]))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-3552913d90d9> in <module>()
    ----> 1 plot_data(np.array([[1,1],[2,2],[3,3]]),np.array([0,1,2]))
    

    NameError: name 'plot_data' is not defined


Example of using `plot_decision_region` defining first a simple mathematical function taking place for a trained prediction function.


```python
X = np.array([[1,1],[2,2],[3,3]])
def mypredictor(x):
  return np.sum(x)
#plot_decision_region(X,mypredictor) 

```

## Import `qmc.tf.layers` and `qmc.tf.models`

Now we import
- `accuracy_score`: For calculating how good a classifier is. Thus counting the good predictions and dividing by the total of elements in the test set.
- `make_blobs, make_moons, make_circles`: Datasets for doing classification.
- `MinMaxScaler, OneHotEncoder`: MinMaxScaler scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one. OneHotEncoder maps a categorical value into binary variables corresponding to the number of classes in the categorical variable.
- `train_test_split`: Spliting the data.
- `tf`: Tensorflow.
- `layers`: Custom layers in qmc.
- `models`: Custom models in qmc.


```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import qmc.tf.layers as layers
import qmc.tf.models as models
```

    /usr/local/lib/python3.7/dist-packages/typeguard/__init__.py:804: UserWarning: no type annotations present -- not typechecking qmc.tf.layers.CrossProduct.__init__
      warn('no type annotations present -- not typechecking {}'.format(function_name(func)))
    /usr/local/lib/python3.7/dist-packages/typeguard/__init__.py:804: UserWarning: no type annotations present -- not typechecking qmc.tf.layers.DensityMatrix2Dist.__init__
      warn('no type annotations present -- not typechecking {}'.format(function_name(func)))
    /usr/local/lib/python3.7/dist-packages/typeguard/__init__.py:804: UserWarning: no type annotations present -- not typechecking qmc.tf.layers.DensityMatrixRegression.__init__
      warn('no type annotations present -- not typechecking {}'.format(function_name(func)))


## Kernel density estimation : Classical approach

We first import two distributions `norm` and `bernoulli`. Then `gaussian_kde` for kernel density estimation using gaussian kernels.

Then we define the class `Mixture` inherited from `object` for defining a two component gaussian mixture

$$
f (x) = \alpha\left(\frac{1}{\sqrt{2\pi\sigma^2}}exp\left(\frac{(x-\mu_1)^2}{2\sigma_1^2}\right)\right) + \left(1-\alpha \right) \left(\frac{1}{\sqrt{2\pi\sigma^2}}exp\left(\frac{(x-\mu_2)^2}{2\sigma_2^2}\right)\right).
$$


Why not only `class Mixture()` ? Read the discussion [here](https://stackoverflow.com/questions/4015417/python-class-inherits-object).

In the constructor `__init__` we define five parameters: four for two univariate normals (`loc1`,`loc2`,`scale1` and `scale2`) + one `alpha` for the weights of the mixture. With the location and scale parameters we define two normals and assign them to `var1` and `var2`.

And two methods
- `pdf` which returns the mixture of two gaussians.
- `rvs` which returns a sample of the mixture. Remember that first choose a mixture component by drawing $j$ from the categorical distribution with probabilities $[\pi_1,\dots,\pi_d]$. This can be done using a random number generator for the categorical distribution. Note. In our case we only have two components so the "categorical distribution" reduces to a bernoulli distribution.

Now we create an object `mixt` from `Mixture` class with some given parameters. That give us a "bimodal" shape curve for the density.

Then we extract a sample of `size = 100` from mixt using the `rvs` method. Save it to `sample`.

In `kernel` we save the $\hat f_{kde}$ estimated using **kernel density estimation** in a classical non parametric fashion applied to `sample` vector. Thus using `gaussian_kde`. Notice `gaussian_kde` includes automatic bandwidth determination.

Finally we plot **mixt** $f$ and **kernel** $\hat f_{kde}$ using `pl.plot` for a windows plot vector `x`.

**Some orientation:**

As we want to compare our quantum way of estimating the density, we have to start out by defining a theoretical baseline model we want to approximate, in our case is `mixt`. Finally, the estimation method we would like to compare our model with is non pararmetric kernel density estimation, so we save this estimation in `kernel`.


```python
from scipy.stats import norm, bernoulli, gaussian_kde

class Mixture(object):

    def __init__(self, loc1=0, scale1=1, loc2=1, scale2=1, alpha=0.5):
        self.var1 = norm(loc=loc1, scale=scale1)
        self.var2 = norm(loc=loc2, scale=scale2)
        self.alpha = alpha
        
    def pdf(self, x):
        return (self.alpha * self.var1.pdf(x) + 
                (1 - self.alpha) * self.var2.pdf(x))

    def rvs(self, size=1):
        vals = np.stack([self.var1.rvs(size), self.var2.rvs(size)],  axis=-1)
        print(vals.shape)
        idx = bernoulli.rvs(1. - self.alpha, size=size)
        return np.array([vals[i, idx[i]] for i in range(size)])

n_var = norm(loc=1, scale=1)
mixt = Mixture(loc1=0, scale1=0.5, loc2=3, scale2=1, alpha=0.5)
sample = mixt.rvs(100) 
kernel = gaussian_kde(sample)
x = np.linspace(-10.,10.,99)
pl.plot(x, mixt.pdf(x), 'r-',  alpha=0.6, label='norm pdf')
pl.plot(x, kernel(x), 'b-',  alpha=0.6, label='kde pdf')

```

    (100, 2)





    [<matplotlib.lines.Line2D at 0x7f33815f2490>]




![png](DMKDE_files/DMKDE_18_2.png)


### How to sample from the Guassian Mixture Model?

First we stack the two random samples from the two gaussian densities in `valstest`.


```python
valstest = np.stack([norm(loc = 0, scale = 1).rvs(size = 4),norm(loc = 10, scale = 1).rvs(size = 4)], axis = -1 )
```


```python
valstest
```




    array([[-0.14509451,  9.85286915],
           [ 0.34545315, 11.22459855],
           [-0.38473518,  9.34825703],
           [-0.88421728,  8.82330827]])



Now we generate the indexes that will help us to choose the component according to `alpha` and the `size` of the sample we want (comes from the last step). Notice that we have two components in the mixture, so our vector will contain either `0` or `1`. We recall that here we can use a bernoulli since it has two possible outcomes, but if we´d had N components to sample from we would use a categorical distribution in this step.


```python
idxtest = bernoulli.rvs(1. - 0.5, size=4)
idxtest
```




    array([0, 1, 1, 1])



Now we choose our desired sample from `valstest` according to the index in `idxtest`.


```python
np.array([valstest[i, idxtest[i]] for i in range(4)])
```




    array([-0.14509451, 11.22459855,  9.34825703,  8.82330827])



## Kernel density estimation: Quantum Measurement approach

---



### Some context on the layers to be used in `QMDensity()`

To run the `dmkde` method for estimating the density $f$ we do some things first.

Let´s remember that we generated a sample from the gaussian mixture model and save it to `sample`. `sample` shape is `(3,)` so we first reshape it to `(3,1)` in order to have a sample with the correct dimensions into our model.

Define a dimension variable `dim = 300` which will be the number of random Fourier features.

Now we create an object feature map for values in the `x` dimension into the variable `fm_x`. Let´s remember that the class `QFeatureMapRFF` is inherited from `tf.keras.Layer` whose initialization parameters are

        input_dim: dimension of the input
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.

And let's recall that in order to generate random fourier features we must 


0. Choose the number of Random Fourier Features $R$. 
1. Draw $R$ samples $\{\mathbf{w}_1,\dots,\mathbf{w}_R\}$ from a **gaussian distribution** $\mathcal{N}(\mathbf{0},\mathbf{I_d})$ since Bochner theorem asserts that $K(\Delta) = \int exp(iw\Delta)p(w)dw$ and we know that under the Fourier transform, the Gaussian function is mapped to another Gaussian function with a different width.
2. Draw $R$ samples $\{b_1,\dots,b_R\}$ from an uniform distribution in $(0,2\pi)$.
3. Our desired random map will be $\phi_{rff}: \mathbb{R}^d \rightarrow \mathbb{R}^R$ such that
$$
\phi_{rff}(\mathbf{x}) = \begin{pmatrix} \frac{1}{\sqrt{R}} \sqrt{2}cos(\mathbf{w_1^Tx }+b) \\ \vdots \\ \frac{1}{\sqrt{R}} \sqrt{2}cos(\mathbf{w_R^Tx }+b)\end{pmatrix} 
$$

Notice that in order to build such map we need to do the following matching between math elements and variables in the constructor 

- `input_dim` : $d$
- `dim` : $R$
- `gamma` : Bandwidth parameter of the internal gaussian distribution ?

Notice that at the end with end up with $\psi = $ `norm (tf.cos(w^tx +b)* tf.sqrt(2. / self.dim) ) / tf.expand_dims(norm(vector), axis=-1)`. ????

Where is the map ???

Understand what `RBFSampler` does ? estimates the internal kernel ? Or the outside kernel ? Why to define such a big layer when you have a direct method that approximates the kernel using `RBFSampler`.

### How does `QFeatureMapRFF` class work ?

Behind the functioning of this layer is the RBF sampler which approximates a Radial Basis Function (the gaussian belongs to this class). From instancing and `RBFSampler` we extract the weights $\mathbf{w_i}$'s and the offsets $b_i$'s and build the Quantum Feature Map.


```python
# d = 1  
from sklearn.kernel_approximation import RBFSampler

rbf_sampler = RBFSampler(
            gamma=1,
            n_components=300,
            random_state=17)
x = np.zeros(shape=(1, 1))
rbf_sampler.fit(x)
rbf_sampler.random_weights_.shape
rbf_sampler.random_offset_.shape
```




    (300,)



What is doing the layer when passing a vector of 300


```python
fm_x = layers.QFeatureMapRFF(1, dim=300, gamma = 1, random_state=17)
fm_x(np.zeros(3).reshape(-1,1)).shape
```




    TensorShape([3, 300])



### How does `QMeasureDensity` layer class work ? 

In this class we compute the value of the density by doing:

$$
\rho = \frac{1}{N} \sum_i^Nz_iz_i^*
$$

where N is the number of samples.

### Training a `QMDensity` model 

Here we just use the `QMDensity` model which has the following structure:

```
class QMDensity(tf.keras.Model):
    """
    A Quantum Measurement Density Estimation model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
    """
    def __init__(self, fm_x, dim_x):
        super(QMDensity, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.qmd = layers.QMeasureDensity(dim_x)
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        return probs

    @tf.function
    def call_train(self, x):
        if not self.qmd.built:
            self.call(x)
        psi = self.fm_x(x)
        rho = self.cp([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x = data
        rho = self.call_train(x)
        self.qmd.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMDensity, self).fit(*args, **kwargs)
        self.qmd.weights[0].assign(self.qmd.weights[0] / self.num_samples)
        return result

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

```

In general it has two layers and they are defined in the `__init__`:

- `fm_x` = Quantum feature map which is a layer of `QFeatureMapFRR`.
- `qmd` = Which is a layer that actually do the measurement which is a layer of type `QMeasureDensity(dim_x)`.


These two layers are concatenated and we can see this model's forward pass inside the `call` method.

Then a customization is made at the level of training the model using auxiliary methods such as `call_train`.


```python
X.dtype
```




    dtype('float64')




```python
X = sample.reshape((-1, 1))
dim = 300
fm_x = layers.QFeatureMapRFF(1, dim=dim, gamma=1, random_state=17)
qmd = models.QMDensity(fm_x, dim)
qmd.compile()
qmd.fit(X, epochs=1)
out = qmd.predict(x.reshape((-1, 1)))
pl.plot(x, mixt.pdf(x), 'r-',  alpha=0.6, label='norm pdf')
pl.plot(x, kernel(x), 'b-',  alpha=0.6, label='kde pdf')
pl.plot(x, out, 'g-',  alpha=0.6, label='qmkde pdf')

```

    4/4 [==============================] - 0s 6ms/step





    [<matplotlib.lines.Line2D at 0x7f3379765510>]




![png](DMKDE_files/DMKDE_40_2.png)


### How does `QMeasureDensityEig` layer class work?

In this class we compute the value of the density by doing **matrix factorizatoin**:

$$
\rho = V^*\Lambda V
$$

After doing the calculation in the step above, WHY ???

### Training a `QMDensitySGD()` model


```python
qmd1 = models.QMDensitySGD(1, dim, num_eig=5, gamma=1, random_state=17)
eig_vals = qmd1.set_rho(qmd.weights[2])
out = qmd1.predict(x.reshape((-1, 1)))
pl.plot(x, mixt.pdf(x), 'r-',  alpha=0.6, label='norm pdf')
pl.plot(x, kernel(x), 'b-',  alpha=0.6, label='kde pdf')
pl.plot(x, out, 'g-',  alpha=0.6, label='qmkde pdf')
```




    [<matplotlib.lines.Line2D at 0x7fb523a09510>]




![png](DMKDE_files/DMKDE_44_1.png)



```python
pl.plot(eig_vals[-5:])
tf.reduce_sum(eig_vals)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=1.0000001>




![png](DMKDE_files/DMKDE_45_1.png)


### Training a `QMDensitySGD()` model without training the RFF map ?


```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
qmd3 = models.QMDensitySGD(1, dim, num_eig=4, gamma=1.5, random_state=17)
qmd3.layers[1].trainable = True
qmd3.layers[0].trainable = False
qmd3.compile(optimizer)
#eig_vals = qmd3.set_rho(qmd.weights[2])
qmd3.fit(X, epochs=20)
out = qmd3.predict(x.reshape((-1, 1)))
pl.plot(x, mixt.pdf(x), 'r-',  alpha=0.6, label='norm pdf')
pl.plot(x, kernel(x), 'b-',  alpha=0.6, label='kde pdf')
pl.plot(x, out, 'g-',  alpha=0.6, label='qmkde pdf')
```

    Epoch 1/20
    4/4 [==============================] - 0s 3ms/step - loss: 164.1226
    Epoch 2/20
    4/4 [==============================] - 0s 3ms/step - loss: 100.6198
    Epoch 3/20
    4/4 [==============================] - 0s 3ms/step - loss: 80.4300
    Epoch 4/20
    4/4 [==============================] - 0s 2ms/step - loss: 70.3795
    Epoch 5/20
    4/4 [==============================] - 0s 3ms/step - loss: 62.3509
    Epoch 6/20
    4/4 [==============================] - 0s 3ms/step - loss: 56.7625
    Epoch 7/20
    4/4 [==============================] - 0s 3ms/step - loss: 53.4480
    Epoch 8/20
    4/4 [==============================] - 0s 3ms/step - loss: 53.8071
    Epoch 9/20
    4/4 [==============================] - 0s 3ms/step - loss: 49.5807
    Epoch 10/20
    4/4 [==============================] - 0s 2ms/step - loss: 48.9573
    Epoch 11/20
    4/4 [==============================] - 0s 2ms/step - loss: 46.4132
    Epoch 12/20
    4/4 [==============================] - 0s 3ms/step - loss: 45.3201
    Epoch 13/20
    4/4 [==============================] - 0s 3ms/step - loss: 43.8438
    Epoch 14/20
    4/4 [==============================] - 0s 3ms/step - loss: 44.8129
    Epoch 15/20
    4/4 [==============================] - 0s 3ms/step - loss: 44.7786
    Epoch 16/20
    4/4 [==============================] - 0s 3ms/step - loss: 43.2091
    Epoch 17/20
    4/4 [==============================] - 0s 4ms/step - loss: 44.3267
    Epoch 18/20
    4/4 [==============================] - 0s 3ms/step - loss: 42.1915
    Epoch 19/20
    4/4 [==============================] - 0s 3ms/step - loss: 42.4475
    Epoch 20/20
    4/4 [==============================] - 0s 3ms/step - loss: 42.8540





    [<matplotlib.lines.Line2D at 0x7fb520dcb250>]




![png](DMKDE_files/DMKDE_47_2.png)



```python
qmd3.layers[0].trainable
```




    False




```python

```
