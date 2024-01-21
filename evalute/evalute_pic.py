import os
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from inception_score import IS
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pic_dir', type=str,default="./data/VITON/test_any.txt")
    parser.add_argument('-t', '--test_dir', type=str, default="./data/VITON/VITON_test/test_img")
    parser.add_argument('-r','--result_dir', type=str, default="./results/TEST_PAIR/pf_b_2l2e-05m_63")
    parser.add_argument('-e','--evalute', type=str, default="ssim")
    opt = parser.parse_args()
    return opt
#psnr
#IS

#ssim
def load_pic(pic_dir,test_dir,result_dir,evalute):
    """
        pic_dir表示图片的txt路径
        test_dir表示测试集的图片路径
        result_dir表示模型输出的图片路径
        """
    ##is
    if evalute == "is":
        is_value,_ = IS(result_dir)
        print("is_value:",is_value)
        return
    # ssim
    ssim_values = []
    psnr_values = []
    with open(pic_dir, 'r') as fp:
        img_names = fp.readlines()
    for img_name in img_names:
        img_name = img_name.split(" ")
        test_pic_path = os.path.join(test_dir, img_name[0])
        result_pic_path = os.path.join(result_dir, img_name[0])
        test_img = io.imread(test_pic_path)
        result_img = io.imread(result_pic_path)
        if evalute == 'ssim':
            ssim_value = ssim(test_img, result_img,channel_axis=-1)
            ssim_values.append(ssim_value)
        if evalute == "psnr":
            psnr_value = psnr(test_img,result_img)
            psnr_values.append(psnr_value)
    if evalute == 'ssim':
        print("ssim_value:",sum(ssim_values)/len(ssim_values))
    if evalute == 'psnr':
        print("psnr_value:",sum(psnr_values)/len(psnr_values))

def main():
    opt = get_opt()
    pic_dir = opt.pic_dir
    test_dir = opt.test_dir
    result_dir = opt.result_dir
    evalute = opt.evalute
    load_pic(pic_dir, test_dir, result_dir,evalute)
    # print(img_names)
if __name__ == '__main__':
    main()