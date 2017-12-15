import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from scipy.misc import imread, imsave

from dataset import load_dataset

def run_pca(im, r, g, b, rank):
    approx = np.zeros_like(im)
    for channel, (U, s, V) in enumerate([r, g, b]):
        for i in range(rank):
            approx[:,:,channel] += s[i] * np.dot(U[:,i:i+1], V[i:i+1,:])
    return np.clip(approx, 0.0, 1.0)

def eval_psnr(im, gt):
    dims = im.shape
    mse = np.sum(np.power(im - gt, 2.0)) / (dims[0] * dims[1])
    return -10 * np.log(mse) / np.log(10.)    

def find_best_rank(im, gt, rank_min=10, rank_max=250, rank_step=10):
    last_psnr = 0
    last_ssim = 0
    last_approx = None
    r = np.linalg.svd(im[:,:,0])
    g = np.linalg.svd(im[:,:,1])
    b = np.linalg.svd(im[:,:,2])
    for rank in range(max(1, rank_min), rank_max+1)[::rank_step]:
        approx = run_pca(im, r, g, b, rank)
        psnr = eval_psnr(approx, gt)
        if psnr < last_psnr:           
            break
        last_psnr = psnr
        last_approx = approx

    print rank-rank_step, last_psnr, ssim(last_approx, gt, multichannel=True)
    return last_approx

def main():
    train, dev, test, loader = load_dataset(False)
    for im, gt in dev:
        #if im[0] == 'lens' and im[1] == 'iso1600' and im[3] in [11, 12]:
        if im == ('cereal2', 'iso102k', 9, 12):
            filename = "output/%s_%s_%s_%s.bmp" % im
            print filename
            result = find_best_rank(loader(im), loader(gt), 1, 300, 1)
            result = np.array(np.clip(result, 0.0, 1.0) * 255, dtype=np.uint8)
            imsave(filename, result)

if __name__ == '__main__':
    main()
