# Adapted from:
# https://github.com/yaohungt/Pointwise_Dependency_Neural_Estimation/blob/master/MI_Est_and_CrossModal/src/estimators.py
# https://github.com/ermongroup/smile-mi-estimator/blob/master/estimators.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense

# ------------------------------------------------------------------------
# Evaluate MI Estimator
# ------------------------------------------------------------------------

def calc_neural_mi(dataset, n_epochs, critic='separable', train_obj='js_fgan', eval_type='smile', print_mi=True, alpha=1.0, clip=None):
    """
    Estimates the mutual information between x and y using variational neural estimator.
    
    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset containing pairs of input data (x, y) for training.
    n_epochs : int
        The number of epochs to train the model.
    critic : str, optional
        The type of critic model to use. Available options are 'concat' and 'separable' [Default is 'separable'].
    train_obj : str, optional
        The training objective to use for estimating MI. Available options are 'nwj', 'dv', 'cpc', 'js_fgan' [Default is 'js_fgan'].
    eval_type : str, optional
        The evaluation method to use for estimating MI. Available options are 'nwj', 'dv', 'cpc', 'smile', 'direct' [Default is 'smile'].
    print_mi : bool, optional
        Whether to print the MI values during training [Default is True].
    alpha : float, optional
        A parameter for the 'smile' estimator to control clipping [Default is 1.0].
    clip : float, optional
        A parameter for the 'smile' estimator to set the clipping threshold [Default is None].
    
    Returns
    -------
    mi : float
        Mutual information between x and y [in nats].
    """
    
    model = train_critic_model(dataset, n_epochs, critic='separable', train_obj='js_fgan', eval_type='direct', print_mi=True)
    
    mi_batch = 0.
    for step, (x_batch, y_batch) in enumerate(dataset):    
        score = model(x_batch, y_batch)
        if eval_type == 'nwj':
            mi_batch += nwj_lower_bound(score)
        elif eval_type == 'dv':
            mi_batch += dv_upper_lower_bound(score)
        elif eval_type == 'cpc':
            mi_batch += cpc_lower_bound(score)
        elif eval_type == 'smile':
            mi_batch += smile_lower_bound(score, alpha, clip)
        elif eval_type == 'direct':
            mi_batch += direct_log_density_ratio(score)
        else:
            raise NotImplementedError(f"Estimator ({estimator}) not supported.")
    mi = mi_batch/len(dataset)
    return mi.numpy()

# ------------------------------------------------------------------------
# Train MI Estimator
# ------------------------------------------------------------------------

def train_critic_model(dataset, n_epochs, critic='separable', train_obj='js_fgan', eval_type='direct', print_mi=True):
    """
    Trains a critic model to estimate mutual information.
    
    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset containing pairs of input data (x, y) for training.
    n_epochs : int
        The number of epochs to train the model.
    critic : str, optional
        The type of critic model to use. Available options are 'concat' and 'separable' [Default is 'separable'].
    train_obj : str, optional
        The training objective to use for estimating MI. Available options are 'nwj', 'dv', 'cpc', 'js_fgan' [Default is 'js_fgan'].
    eval_type : str, optional
        The evaluation method to use for estimating MI. Available options are 'nwj', 'dv', 'cpc', 'smile', 'direct' [Default is 'direct'].
    print_mi : bool, optional
        Whether to print the MI values during training [Default is True].

    
    Returns
    -------
    model : tf.keras.Model
        The trained critic model.
    all_mi_train : np.ndarray
        Array of shape (n_epochs,) containing MI values estimated using the train_obj.
    all_mi_eval : np.ndarray
        Array of shape (n_epochs,) containing MI values estimated using the eval_type.
    """
    
    if critic == 'concat':
        model = ConcatCritic(dataset)
    elif critic == 'separable':
        model = SeparableCritic(dataset)
    else:
        raise NotImplementedError(f"Critic model ({critic}) not supported.")
    
    if train_obj == 'nwj':
        loss_fn = nwj_lower_bound
    elif train_obj == 'dv':
        loss_fn = dv_upper_lower_bound
    elif train_obj == 'cpc':
        loss_fn = cpc_lower_bound
    elif train_obj =='js_fgan':
        loss_fn = js_fgan_lower_bound
    else:
        raise NotImplementedError(f"Estimator ({train_obj}) not supported.")
        
    if eval_type == 'nwj':
        estimator = nwj_lower_bound
    elif eval_type ==  'dv':
        estimator = dv_upper_lower_bound
    elif eval_type == 'cpc':
        estimator = cpc_lower_bound
    elif eval_type == 'smile':
        estimator = smile_lower_bound
    elif eval_type == 'direct':
        estimator = direct_log_density_ratio
    else:
        raise NotImplementedError(f"Estimator ({eval_type}) not supported.")
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    @tf.function
    def train_step(x, y, model, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            score = model(x,y)
            loss_value = -loss_fn(score)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return -loss_value
    
    @tf.function
    def eval_step(x, y, model, estimator):
        score = model(x,y)
        mi = estimator(score)
        return mi
    
    all_mi_train = []
    all_mi_eval = []
    for epoch in range(n_epochs):
        mi_train = 0.
        mi_eval = 0.
        for step, (x_batch, y_batch) in enumerate(dataset):
            mi_train += train_step(x_batch, y_batch, model, optimizer, loss_fn)
            mi_eval += eval_step(x_batch, y_batch, model, estimator)
        all_mi_train.append(mi_train/len(dataset))
        all_mi_eval.append(mi_eval/len(dataset))
        if print_mi:
            print(f'Epoch f"{epoch+1:03}/{n_epochs}: MI (train)={all_mi_train[-1]:.3f}, MI (eval)={all_mi_eval[-1]:.3f}')
    
    all_mi_train = np.array(all_mi_train)
    all_mi_eval = np.array(all_mi_eval)
    return model

# ------------------------------------------------------------------------
# Estimator for Lower Bound for MI
# ------------------------------------------------------------------------

def nwj_lower_bound(score):
    """
    Estimates the mutual information using MINE-f (Belghazi et al., 2018) based on NWJ lower bound.
    
    Parameters
    ----------
    score : tf.Tensor
        A square matrix of scores representing the joint distribution. The diagonal elements 
        correspond to joint samples (x, y), and the off-diagonal elements correspond to 
        marginal samples.
        
    Returns
    -------
    tf.Tensor
        The estimated mutual information.
    
    """
    
    batch_size = tf.shape(score)[0]
    
    joint_term = tf.reduce_mean(tf.linalg.diag_part(score))
    marg_term = tf.exp(logmeanexp_nodiag(score))
    
    return joint_term - marg_term

def dv_upper_lower_bound(score):
    """
    Estimates the mutual information using based on DV lower bound, but upper bounded by using log outside.
    
    Parameters
    ----------
    score : tf.Tensor
        A square matrix of scores representing the joint distribution. The diagonal elements 
        correspond to joint samples (x, y), and the off-diagonal elements correspond to 
        marginal samples.
        
    Returns
    -------
    tf.Tensor
        The estimated mutual information.
    
    """
    
    first_term = tf.reduce_mean(tf.linalg.diag_part(score))
    second_term = logmeanexp_nodiag(score)
    
    return first_term - second_term

def cpc_lower_bound(score):
    """
    Estimates the mutual information using CPC (van den Oord et al., 2018).
    
    Parameters
    ----------
    score : tf.Tensor
        A square matrix of scores representing the joint distribution. The diagonal elements 
        correspond to joint samples (x, y), and the off-diagonal elements correspond to 
        marginal samples.
        
    Returns
    -------
    tf.Tensor
        The estimated mutual information.
    
    """
    
    diag_mean = tf.reduce_mean(tf.linalg.diag_part(score))
    logsumexp = tf.reduce_logsumexp(score, axis=1)
    nll = diag_mean - logsumexp
    batch_size = tf.cast(tf.shape(score)[0], tf.float32)

    return tf.reduce_mean(tf.math.log(batch_size) + nll)

def js_fgan_lower_bound(score):
    """
    Estimates the mutual information using JS F-GAN lower bound (Poole et al., 2019).
    
    Parameters
    ----------
    score : tf.Tensor
        A square matrix of scores representing the joint distribution. The diagonal elements 
        correspond to joint samples (x, y), and the off-diagonal elements correspond to 
        marginal samples.
        
    Returns
    -------
    tf.Tensor
        The estimated mutual information.
    
    """
    
    score_diag = tf.linalg.diag_part(score)
    first_term = -tf.reduce_mean(tf.nn.softplus(-score_diag))
    batch_size = score.shape[0]
    second_term = (tf.reduce_sum(tf.nn.softplus(score)) - tf.reduce_sum(tf.nn.softplus(score_diag))) / (batch_size * (batch_size - 1.))
    
    return first_term - second_term

def smile_lower_bound(score, alpha=1.0, clip=None):
    """
    Estimates the mutual information using SMILE (Song and Ermon, 2020).
    
    Parameters
    ----------
    score : tf.Tensor
        A square matrix of scores representing the joint distribution. The diagonal elements 
        correspond to joint samples (x, y), and the off-diagonal elements correspond to 
        marginal samples.
    alpha : float, optional
        A scaling factor applied to the score [Default is 1.0].
    clip : float, optional
        The clipping threshold for the score. If None, no clipping is applied [Default is None].
        
    Returns
    -------
    tf.Tensor
        The estimated mutual information.
    
    """
    
    if clip is not None:
        score = tf.clip_by_value(score * alpha, -clip, clip)
    z = logmeanexp_nodiag(score * alpha)
    dv_clip = tf.reduce_mean(tf.linalg.diag_part(score)) - z

    return dv_clip

def direct_log_density_ratio(score):
    """
    Estimates the mutual information using the direct log density ratio from the given score matrix.
    
    Parameters
    ----------
    score : tf.Tensor
        A square matrix of scores representing the joint distribution. The diagonal elements 
        correspond to joint samples (x, y), and the off-diagonal elements correspond to 
        marginal samples.
        
    Returns
    -------
    tf.Tensor
        The estimated mutual information.
    
    """
    
    return tf.reduce_mean(tf.linalg.diag_part(score))

# ------------------------------------------------------------------------
# Critic architectures
# ------------------------------------------------------------------------

def mlp_critic(input_dim, output_dim):
    """
    Create a multi-layer perceptron (MLP) critic model.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input data.
    output_dim : int
        The dimensionality of the output data.

    Returns
    -------
    tf.keras.Model
        The constructed MLP model.
        
    """
    
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=(input_dim,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(output_dim))
    return model

class SeparableCritic(tf.keras.Model):
    """
    A separable critic model for estimating mutual information.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset containing pairs of input data (x, y).
    output_dim : int
        The output dimension of the critic networks g and h [Default is 128].

    Methods
    -------
    call(x, y):
        Computes the score for the input data by passing x through network g and y through network h, 
        and then computing the dot product of the outputs.

    get_config():
        Returns the configuration of the model, including the output dimension and configurations of the 
        neural networks g and h.
        
    """
    
    # pass x to g and pass y to h --> f(x,y) = g(x)^T h(y) --> only require 2N forward passes
    def __init__(self, dataset, output_dim=128, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        dim_x = dataset.element_spec[0].shape[1]
        dim_y = dataset.element_spec[1].shape[1]
        self.output_dim = output_dim
        self._g = mlp_critic(dim_x, self.output_dim)
        self._h = mlp_critic(dim_y, self.output_dim)
    def call(self, x, y):
        g_output = self._g(x)
        h_output = self._h(y)
        score = tf.matmul(h_output, tf.transpose(g_output))
        return score   # shape = (batch_size, batch_size)
    def get_config(self):
        config = super(SeparableCritic, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'g': self._g.get_config(),
            'h': self._h.get_config()
        })
        return config

class ConcatCritic(tf.keras.Model):
    """
    A concatenated critic model for estimating mutual information.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset containing pairs of input data (x, y).

    Methods
    -------
    call(x, y):
        Computes the scores for the concatenated input tensors x and y.

    get_config():
        Returns the configuration of the model.
        
    """
    
    # concatenate x and y --> require batch_size^2 forward passes
    def __init__(self, dataset, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        dim_x = dataset.element_spec[0].shape[1]
        dim_y = dataset.element_spec[1].shape[1]
        self._f = mlp_critic(dim_x+dim_y, 1)  # output is scalar score
    def call(self, x, y):
        # shape of x: (batch_size, dim_x)
        # shape of y: (batch_size, dim_y)
        batch_size = tf.shape(x)[0]
        x_tiled = tf.tile(tf.expand_dims(x, axis=1), [1, batch_size, 1])    # shape = (batch_size, batch_size, dim_x)
        y_tiled = tf.tile(tf.expand_dims(y, axis=0), [batch_size, 1, 1])    # shape = (batch_size, batch_size, dim_y)
        y_tiled = tf.cast(y_tiled, dtype=x.dtype)
        xy_pairs = tf.concat([x_tiled, y_tiled], axis=-1)                   # shape = (batch_size, batch_size, dim_x+dim_y)
        score = self._f(tf.reshape(xy_pairs, [batch_size * batch_size, -1]))
        return tf.reshape(score, [batch_size, batch_size])                 # shape = (batch_size, batch_size)
    def get_config(self):
        config = super(ConcatCritic, self).get_config()
        config['f'] = self._f
        return config

# ------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------

def logmeanexp_nodiag(x, axis=None):
    """
    Compute the log-mean-exponential of input tensor x, excluding the diagonal elements.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    axis : tuple, optional
        The dimensions over which to compute the log-mean-exponential [Default is None, which computes over all dimensions].

    Returns
    -------
    tf.Tensor
        The log-mean-exponential of the input tensor, excluding the diagonal elements.
    
    """
    
    batch_size = tf.shape(x)[0]
    if axis is None:
        axis = (0, 1)

    # Create a mask to set diagonal elements to -inf
    inf_mask = tf.linalg.diag(tf.fill([batch_size], np.inf))
    x_no_diag = x - inf_mask

    # Compute logsumexp excluding diagonal
    logsumexp = tf.reduce_logsumexp(x_no_diag, axis=axis)

    # Compute the number of elements excluding the diagonal
    if isinstance(axis, (tuple, list)) and len(axis) == 1:
        num_elem = tf.cast(batch_size - 1, tf.float32)
    else:
        num_elem = tf.cast(batch_size * (batch_size - 1), tf.float32)

    return logsumexp - tf.math.log(num_elem)
