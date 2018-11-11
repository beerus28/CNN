import numpy as np


def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    mean = np.mean(x, axis=1) # calculates the mean of the array x
    mean = mean[:, np.newaxis]
    x = np.subtract(x, mean) # this is euivalent to subtracting the mean of x from each value in x
    
    return x


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    mean = np.mean(x, axis=1) # calculates the mean of the array x
    mean = mean[:, np.newaxis]
    print(x.shape)
    x_sqrd = np.square(np.subtract(x, mean)) # this is euivalent to subtracting the mean of x from each value in x
    x_summed = np.sum(x_sqrd, axis=1)
    
    sigma = np.divide(x_summed, x.shape[1])
    dem = np.sqrt(bias + sigma)
    print(sigma.shape)
    x_sub = scale*x
    sigma = sigma[:, np.newaxis]
    print(dem.shape)
    dem = dem[:, np.newaxis]
    
    x = x_sub / dem
    
    return x


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    mean = np.mean(x, axis=0) # calculates the mean of the array x
    #mean = mean[:, np.newaxis]
    print(mean.shape)
    
    x = np.subtract(x, mean) # this is euivalent to subtracting the mean of x from each value in x
    
    xtest = np.subtract(xtest, mean)
    
    print(x.shape)
    return (x, xtest)


def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    id_size = x.shape[0]
    id_size2 = x.shape[1]
    print(id_size)
    U, S, V = np.linalg.svd(np.dot(x.T, x) / id_size + np.eye(id_size2, dtype=float)*bias)
    
    
    pca = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S))), U.T)

    whitened_x = np.dot(x, pca)
    whitened_xtest = np.dot(xtest, pca)
    
    # U,S,V = svd( x.T dot x /n + eye*bias)
    # pca = U dot diag(1. / sqrt(S)) dot U.T
    # zca x = x dot pca

    return (whitened_x, whitened_xtest)


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    meaned_x = sample_zero_mean(x)
    meaned_xtest = sample_zero_mean(xtest)
    
    gcn_x = gcn(meaned_x)    # pass sample mean zero
    gcn_xest = gcn(meaned_xtest)
    
    feat_meaned_x, feat_meaned_xtest = feature_zero_mean(gcn_x, gcn_xest)   
    
    whitened_x, whitened_xtest = zca(feat_meaned_x, feat_meaned_xtest)   # assumes zero mean
    print(whitened_x.shape, whitened_xtest)
    whitened_x = whitened_x.reshape(whitened_x.shape[0], 3, image_size, image_size)
    whitened_xtest = whitened_xtest.reshape(whitened_xtest.shape[0], 3, image_size, image_size)
    
    return (whitened_x, whitened_xtest)