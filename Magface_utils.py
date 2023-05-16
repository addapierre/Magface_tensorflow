import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter
from sklearn.metrics import silhouette_score
from math import pi
from tqdm import tqdm

def pairwise_cosine_similarity(embeddings, norm : bool = True):
    """Compute the 2D matrix of cosine similarity between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        tensor of shape (batch_size, batch_size)
    """
    if norm:
        embeddings = tf.math.l2_normalize(embeddings, axis = 1)
    return 1 - tf.matmul(embeddings, embeddings, transpose_b=True)


class MagfaceMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_images, gt, n_step=0):
        """
        Validation metric callback for arcface.
        Arg: 
          val_images: Array containing the already preprocessed validation images.
          gt: ground truth. array containing the labels corresponding to the validation images.
          n_step: integer. validation is performed every n_step batches. 
            used if the user want to monitor the validation during an epoch. 
            default: 0 => validation only at the end of epochs.
        """
        super(MagfaceMetricsCallback, self).__init__()
        self.val_img = val_images
        self.labels = gt
        self.n_step = n_step

        # embeddings are computed batch-wise
        batch_size = 50
        num_batches = np.ceil(self.labels.shape[0] / batch_size)
        self.batch_idx = np.array_split(range(self.labels.shape[0]), num_batches)

    def make_embeddings(self):
        imgs = self.val_img[self.batch_idx[0]]
        # model returns Maglayer classifier output, feature norms and unnormalized embedding.
        _, _, embeddings = self.model(imgs)
        for batch in tqdm(self.batch_idx[1:]):
            imgs = self.val_img[batch]
            _, _, M = self.model(imgs)
            embeddings = tf.concat([embeddings, M], 0)
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        return embeddings
    
    
    def get_accuracy(self, labels, embeddings):
      
      """
      Computes rank-N accuracy for deep metric learning.
      Args:
          labels: array of shape (batch_size,). Labels associated to the embeddings
          embeddings: tensor of shape (batch_size, embed_dim)
      Returns:
          (float, float, float) rank 1, 5, and 10 accuracies
      """
      labels = tf.squeeze(labels)
      pairwise_dist = pairwise_cosine_similarity(embeddings)

      # We need to add a mask to the pairwise_dist tensor, so the lowest distance is not the embedding with itself.
      max_dist = tf.reduce_max(pairwise_dist)
      mask = tf.cast(tf.eye(tf.shape(labels)[0]), tf.float32)
      mask *= max_dist
      # the cast in tf.float64 is to prevent strange behavior during the argsort, likely caused by arm64 architecture in graph mode.
      pairwise_dist_masked = tf.cast(pairwise_dist+mask, tf.float64)

      #argsort: select the 10 lowest distances' position for each line
      result = tf.argsort(pairwise_dist_masked, axis = 1, direction="ASCENDING")[:,:10]

      # seek labels corresponding to those distances with tf.gather
      result = tf.map_fn(lambda x: tf.gather(params = tf.squeeze(labels), indices=x), result, fn_output_signature = tf.int64)
      result = tf.transpose(result, (1,0))
      # compare them to the label array, returns 1 or 0
      accuracy = tf.equal(result, labels)
      # cast back to tf.float32
      accuracy = tf.cast(tf.transpose(accuracy, (1,0)), tf.float32)
      accuracy1 = tf.reduce_mean(tf.reduce_max(accuracy[:,:1], axis = 1))
      accuracy5 = tf.reduce_mean(tf.reduce_max(accuracy[:,:5], axis = 1))
      accuracy10 = tf.reduce_mean(tf.reduce_max(accuracy, axis = 1) )    
      return accuracy1, accuracy5, accuracy10
    
    def on_train_batch_end(self, batch, logs = None):
        if self.n_step != 0:
            if batch%self.n_step==0:

                embeddings = self.make_embeddings()
                
                accuracy1, accuracy5, accuracy10 = self.get_accuracy(self.labels, embeddings)
                logs["val_acc1"] = accuracy1
                logs["val_acc5"] = accuracy5
                logs["val_acc10"] = accuracy10

                #silhouette score is calculated using the sklearn function
                logs["silhouette_score"] = silhouette_score(embeddings, self.labels)

    def on_epoch_end(self, epoch, logs = None):
        
        embeddings = self.make_embeddings()
        accuracy1, accuracy5, accuracy10 = self.get_accuracy(self.labels, embeddings)
        logs["val_acc1"] = accuracy1
        logs["val_acc5"] = accuracy5
        logs["val_acc10"] = accuracy10
        logs["silhouette_score"] = silhouette_score(embeddings, self.labels)
        print("*** validation: ***")
        print(f'acc1: {logs["val_acc1"]}, acc5 : {logs["val_acc5"]}, acc10 : {logs["val_acc10"]}, silhouette : {logs["silhouette_score"]}')



class MagLayer(keras.layers.Layer):
    """Custom layer for MagFace.
    Input:
        units: number of classification neurons
        args: dict with model hyperparameters
            'scale': scale for arcmargin loss
            'l_margin': lower bound for the margin
            'u_margin': upper bound for the margin
            'l_a': lower bound for the feature's norm
            'u_a': upper bound for the feature's norm
            'lambda_g': the lambda for function g   
    Output:
        cos_theta, x_norm: tensors of shape (batch_size, units), (batch_size, 1)
    
    """

    def __init__(self, units, args, kernel_regularizer=None, **kwargs):
        super(MagLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer

        self.l_a = args['l_a']
        self.u_a = args['u_a']
        



    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-1], self.units],
                                      dtype=tf.float32,
                                      initializer=keras.initializers.HeNormal(),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='kernel')
        self.built = True

    @tf.function
    def call(self, inputs):
        # weight normalization
        weights = tf.math.l2_normalize(self.kernel, axis=0)
        
        
        x_norm = tf.norm(inputs, axis = 1, keepdims=True)
        x_norm = tf.clip_by_value(x_norm, self.l_a, self.u_a)

        # Features are not L2 normalized prior to MagLayer, unlike ArcLayer
        inputs = tf.math.l2_normalize(inputs, axis=1)
        cos_theta = tf.matmul(inputs, weights)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)



        return cos_theta, x_norm

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units,
                       "kernel_regularizer": self.kernel_regularizer})
        return config
    

class MagLoss(keras.losses.Loss):
    """magface loss.
    Input:
        args: dict with model hyperparameters
            'scale': scale for arcmargin loss
            'l_margin': lower bound for the margin
            'u_margin': upper bound for the margin
            'l_a': lower bound for the feature's norm
            'u_a': upper bound for the feature's norm
            'lambda_g': the lambda for function g 
    """

    def __init__(self, args, easy_margin = True, name="magloss"):
        """Build an additive angular margin loss object for Keras model."""
        super().__init__(name=name)

        self.l_margin = args['l_margin']
        self.u_margin = args['u_margin']
        self.l_a = args['l_a']
        self.u_a = args['u_a']
        self.lambda_g = args['lambda_g']
        self.scale = args['scale']
        self.easy_margin = easy_margin

        self.cut_off = np.cos(np.pi/2-self.l_margin)

    def calc_loss_G_margin(self, x_norm):
        #returns g and adaptative margin, which depend both on feature norm
        g = x_norm / (self.u_a**2) + 1/(x_norm)
        g = tf.reduce_mean(g)
        ada_margin = (self.u_margin-self.l_margin) / (self.u_a-self.l_a) * (x_norm-self.l_a) + self.l_margin
        return g, ada_margin


    @tf.function
    def call(self, y_true, cos_theta, x_norm):
                
        # adaptative margin and g regulizer calculation:
        loss_g, ada_margin = self.calc_loss_G_margin(x_norm)

        cos_m, sin_m = tf.math.cos(ada_margin), tf.math.sin(ada_margin)
        sin_theta = tf.math.sqrt(1 - tf.math.square(cos_theta))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        
        if self.easy_margin:
            cos_theta_m = tf.where(cos_theta > 0, 
                                   cos_theta_m, 
                                   cos_theta)
        else:
            mm = tf.math.sin( pi - ada_margin) * ada_margin
            threshold = tf.math.cos( pi - ada_margin)
            cos_theta_m = tf.where(cos_theta > threshold, 
                                   cos_theta_m, 
                                   cos_theta - mm)
        
        
        # safe_margin = sin_m * ada_margin
        # threshold = tf.math.cos(pi - ada_margin)
        # cos_theta_m = tf.where(cos_theta > threshold,
        #                         cos_theta * cos_m - sin_theta * sin_m,
        #                         cos_theta - safe_margin)

        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        # y_true is already one-hot encoded 
        mask = y_true
        cos_t_onehot = cos_theta * mask
        cos_t_margin_onehot = cos_theta_m * mask
        logits = (cos_theta + cos_t_margin_onehot - cos_t_onehot)
        

        
        losses = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, logits)) + self.lambda_g * loss_g
            
        return losses

    def get_config(self):
        config = super(MagLoss, self).get_config()
        config.update({"scale": self.scale})
        return config
    

class MagFaceModel(keras.Model):
    """
    Custom model for MagFace.
    Input:
        n_classes: int. Number of classes in the training dataset
        args: dict with model hyperparameters
            'scale': scale for arcmargin loss
            'l_margin': lower bound for the margin
            'u_margin': upper bound for the margin
            'l_a': lower bound for the feature's norm
            'u_a': upper bound for the feature's norm
            'lambda_g': the lambda for function g 
        backbone: keras model (not called).
        input shape: tuple (height, width, channel)
        fine_tune_layer: int, number of backbone layers to be fine tuned. -1 => all backbone trainable
        embedding_size: embedding size
    Output: 
        Classifier output tensor of shape (batch_size, units),
        feature norm tensor of shape (batch_size, 1), 
        raw embedding tensor of shape  (batch_size, embedding_size)
    """
    def compile(self, optimizer, loss):

        super().compile(optimizer)
        self.loss = loss


    def __init__(self, n_classes : int, args : dict, backbone = tf.keras.application.EfficientNetV2S, input_shape : tuple = (380, 380, 3), fine_tune_layers : int = 0, embedding_size : int = 512):
        
        super(MagFaceModel, self).__init__()
        self.args = args

        # make backbone
        
        self.backbone = backbone(include_top=False, input_shape = input_shape)
        if fine_tune_layers == -1:
            self.backbone.trainable = True
        else:
            self.backbone.trainable = False
            if fine_tune_layers:
                for layer in self.backbone.layers[-fine_tune_layers:]:
                    if not isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = True


        self.flatten = keras.layers.Flatten()
        self.batchnorm1 = keras.layers.BatchNormalization()


        self.embedding_layer = keras.layers.Dense(512)
        self.batchnorm2 = keras.layers.BatchNormalization(name = "embedding")
        self.mag_layer = MagLayer(units=n_classes, args = self.args, name = "mag_layer")


    @tf.function
    def call(self, input_tensor, training = False):
        """
        Returns classifaction output, embedding norms and unnormalized embeddings
        """
        # forward pass:
        x = self.backbone(input_tensor)
        x = self.flatten(x)
        x = self.batchnorm1(x)
        x = self.embedding_layer(x)
        embeddings = self.batchnorm2(x)
        cos_theta, x_norm = self.mag_layer(embeddings)
        return cos_theta, x_norm, embeddings


    @tf.function
    def train_step(self, iterator):

        #data = data_adapter.expand_1d(iterator)

        input_tensor, y_true, _ = data_adapter.unpack_x_y_sample_weight(iterator)

        with tf.GradientTape() as tape:
            cos_theta, x_norm, _ = self(input_tensor, training = True)
            loss_value = self.loss.call(y_true = y_true, cos_theta = cos_theta, x_norm = x_norm)
        
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"MagLoss": loss_value}


