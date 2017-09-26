# -*- coding: utf-8 -*-

"""
lmgist.py

Python implementatino of the GIST descriptor.

Modeling the shape of the scene: a holistic representation of the spatial envelope
Aude Oliva, Antonio Torralba
International Journal of Computer Vision, Vol. 42(3): 145-175, 2001.

https://raw.githubusercontent.com/yskmt/sccomp-app/master/src/lmgist.py
"""

from pdb import set_trace as st

import numpy as np
import numpy.matlib as matlib
from skimage import io, transform, util
import matplotlib.pyplot as plt
from matplotlib import cm

def lmgist(img_raw, param, save_dir=None):
    """
    Assume that there is only one file
    
    """
    
    param.G = create_gabor(param.orientations_per_scale,
                           param.img_size+2*param.boundary_extension)

    n_features = param.G.shape[2]*param.number_blocks**2

    # get gist for different color or not
    if param.color == 1:
        n_colors = 3
    else:
        n_colors = 1
    
    # compute gist features for all scenes
    gist = np.zeros([n_colors, n_features])
    g = []

    # TODO: if gist has already computed, just read the file
    
    for n in range(n_colors):

        # convert to grayscale (not a true grayscale)
        if param.color == 0:
            img = np.mean(img_raw, axis=2)
        # use the specified color
        else:
            img = img_raw[:, :, n]

        # resize and crop image to make it square
        img = transform.resize(img, (param.img_size, param.img_size),
                               clip=True, order=1)
        # scale intensities in range [0, 255]
        img = img-np.min(img)
        img = 255*img/np.max(img)

        # prefiltering
        output = prefilt(img, param.fc_prefilt)

        # get gist
        g = gist_gabor(output, param)
        gist[n, :] = np.reshape(g, len(g))

    # TODO: save gist if a output file is provided
        
    return gist.reshape(n_colors*len(g)), param


def prefilt(img, fc=4):
    """
    assume greyscale (c==1), individual image(N==1)
    """
    
    w = 5
    s1 = fc/np.sqrt(np.log(2.0))

    # pad images to reduce boundary artifacts
    img = np.log(img+1)
    img = util.pad(img, w, mode='symmetric')
    sn, sm = img.shape

    n = max(sn, sm)
    n += np.mod(n, 2)
    img = util.pad(img, ((0, n-sn), (0, n-sm)), mode='symmetric')

    # filter
    fx, fy = np.meshgrid(np.arange(-n/2, n/2), np.arange(-n/2, n/2))
    gf = np.fft.fftshift(np.exp(-(fx**2+fy**2)/(s1**2)))

    # whitening
    output = img - np.real(np.fft.ifft2(np.fft.fft2(img)*gf))

    # local contrast normalization
    localstd = np.sqrt(
        np.abs(np.fft.ifft2(np.fft.fft2(output**2)*gf)))
    output = output / (0.2+localstd)
    
    # crop output to have the same size as the input
    output = output[w:sn-w, w:sm-w]

    return output


def gist_gabor(img, param):
    """
    Assume single image
    Assume greyscale image
    """
    
    w = param.number_blocks
    G = param.G
    be = param.boundary_extension

    n_rows, n_cols = img.shape
    c = 1
    N = c
    
    ny, nx, n_filters = G.shape
    W = w*w
    g = np.zeros((W*n_filters, N))

    # pad image
    img = util.pad(img, be, mode='symmetric')

    img = np.fft.fft2(img)
    k = 0
    for n in range(n_filters):
        ig = np.abs(np.fft.ifft2(img*G[:,:,n]))
        ig = ig[be:ny-be, be:nx-be]
    
        v = downN(ig, w)
        g[k:k+W] = np.reshape(v.T, [W, N])
        k = k + W
        
    return g


def downN(x, N):
    """
    Average over non-overlapping square image blocks
    """

    nx = np.linspace(0, x.shape[0], N+1, dtype=int)
    ny = np.linspace(0, x.shape[1], N+1, dtype=int)
    y = np.zeros((N, N))

    for xx in range(N):
        for yy in range(N):
            v = np.mean(x[nx[xx]+1:nx[xx+1], ny[yy]+1:ny[yy+1]])
            y[xx, yy] = v

    return y


def create_gabor(orientations_per_scale, n):
    """
    Precomputes filter transfer functions. All computations are done on the
    Fourier domain. 

    If you call this function without output arguments it will show the
    tiling of the Fourier domain.
    
    Input
    numberOfOrientationsPerScale = vector that contains the number of
    orientations at each scale (from HF to BF)
    n = imagesize = [nrows ncols] 
    
    output
    G = transfer functions for a jet of gabor filters 
    """

    n_scales = len(orientations_per_scale)
    n_filters = sum(orientations_per_scale)

    n = [n, n]
    l = 0
    param = []
    for i in range(n_scales):
        for j in range(orientations_per_scale[i]):
            l += 1
            param.append([0.35,
                          0.3/(1.85**i),
                          16*orientations_per_scale[i]**2/32.0**2,
                          np.pi/orientations_per_scale[i]*j])
            
    # frequencies
    [fx, fy] = np.meshgrid(np.arange(-n[1]/2, n[1]/2),
                           np.arange(-n[0]/2, n[0]/2))
    fr = np.fft.fftshift(np.sqrt(fx**2+fy**2))
    t = np.fft.fftshift(np.angle(fx+1j*fy))

    # transfer functions
    G = np.zeros((n[0], n[1], n_filters))
    for i in range(n_filters):
        tr = t+param[i][3]
        tr = tr+2*np.pi*(tr<(-np.pi)) - 2*np.pi*(tr>np.pi)

        G[:, :, i] = np.exp(-10*param[i][0]*(fr/n[1]/param[i][1]-1)**2\
                            -2*param[i][2]*np.pi*tr**2)

    return G


def show_gist(gist, param):

    n_dim = gist.shape[1]
    nx = ny = 1

    n_blocks = param.number_blocks
    n_filters = np.sum(param.orientations_per_scale)
    n_scales = len(param.orientations_per_scale)

    if hasattr(param, 'G') is False:
        param.G = create_gabor(param.orientations_per_scale,
                               param.img_size+2*param.boundary_extension)
    
    C = cm.hsv(np.arange(0, 256, int(256/n_scales)))[:, :-1]
    colors = np.zeros([n_filters, 3])

    ct = 0
    for s in range(n_scales):
        colors[ct:ct+param.orientations_per_scale[s], :]\
            = matlib.repmat(C[s, :], param.orientations_per_scale[s], 1)
        ct += param.orientations_per_scale[s]
    colors = colors.T

    n_rows, n_cols, n_filters = param.G.shape
    n_features = n_blocks**2*n_filters

    if n_dim != n_features:
        raise ValueError('Missmatch between gist descriptors and the parameters')

    G = param.G[::2, ::2, :]
    n_rows, n_cols, n_filters = G.shape
    G += np.fliplr(np.flipud(G))
    G = np.reshape(G, [n_cols*n_rows, n_filters], order='F')

    # plot
    g = np.reshape(gist, [n_blocks, n_blocks, n_filters], order='F')
    g = np.transpose(g, [1, 0, 2])
    g = np.reshape(g, [n_blocks*n_blocks, n_filters], order='F')

    mosaic = np.zeros([G.shape[0], 3, g.shape[0]])
    for c in range(3):
        mosaic[:, c, :] = np.dot(G, (matlib.repmat(colors[c,:], n_blocks**2, 1)*g).T)

    mosaic = np.reshape(mosaic, [n_rows, n_cols, 3, n_blocks*n_blocks],
                        order='F')
    mosaic = np.fft.fftshift(np.fft.fftshift(mosaic, 0), 1)
    mosaic = np.array(mosaic/np.max(mosaic)*255, dtype=np.uint8)
    mosaic[0,:,:,:] = 255
    mosaic[-1,:,:,:] = 255
    mosaic[:,0,:,:] = 255
    mosaic[:,-1,:,:] = 255

    mosaic_plt = np.zeros(
        (mosaic.shape[0]*n_blocks, mosaic.shape[1]*n_blocks, 3), dtype=np.uint8)

    # merge all the mosaics
    for i in range(n_blocks):
        for j in range(n_blocks):
            mosaic_plt[i*n_rows:(i+1)*n_rows, j*n_cols:(j+1)*n_cols, :]\
                = mosaic[:,:,:, i*n_blocks+j]
    plt.imshow(mosaic_plt)
    plt.axis('off')
    plt.show()

    return
        

class param_gist:
    def __init__(self):
        # default parameters
        self.img_size = 128
        self.orientations_per_scale = [8, 8, 8, 8]
        self.number_blocks = 3
        self.fc_prefilt = 4
        # number of pixels to pad
        self.boundary_extension = 32
        self.color = 0

        
DEBUG = False

if DEBUG:
    param = param_gist()
    param.img_size = 256
    param.orientations_per_scale = [8, 8, 8, 8]
    param.number_blocks = 4
    param.fc_prefilt = 4
 
    import matplotlib.image as mpimg
    
    img_file = '../../../example/lena.png'
    img = np.array(mpimg.imread(img_file))    
    gist, param = lmgist(img, param)
    print (gist.shape)
    #show_gist(gist, param)
    print (np.min(gist), np.max(gist))