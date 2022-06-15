# 生成图
import sys, os
import random
from PIL import Image, ImageFont, ImageDraw
import numpy as np 
from tqdm import tqdm
import cv2 


def get_list(path,type):
    list_out = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith(type)])
    return list_out

def makesure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def random_A():
    return str(np.random.randint(0, 1000))

def random_AB():
    num1 = np.random.randint(0, 100)
    num2 = np.random.randint(0, 100)
    sign = random.choice(['+','-','×','÷'])
    out = '%d%s%d' % (num1, sign, num2)

    return out

def random_content():
    ctype = random.choice(['A', 'AB'])
    if ctype == 'A':
        return random_A()
    if ctype == 'AB':
        return random_AB()

def filter_op(op):
    threshold = 150 
    op_data = np.array(op)
    # imgcat(op_data)
    mask = np.where(op_data>threshold)
    nomask = np.where(op_data<=threshold)
    op_data[mask] = 255
    k = 255.0/threshold
    op_data[nomask] = op_data[nomask]*k
    op_data = np.uint8(op_data)

    return op_data


def paste_op(bg, op):
    op_data = filter_op(op)
    bg_data = np.array(bg)
    op_h, op_w = op_data.shape[0], op_data.shape[1]
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]
    
    # resize
    new_w = int(64 * op_w/op_h)
    op_data = cv2.resize(op_data, (new_w, 64) ,interpolation=cv2.INTER_AREA)
    
    # 中心对齐
    x = int(bg_w//2-new_w//2)
    bg_center = bg_data[:, x:x+new_w,:]

    # 对应相乘
    bg_center = bg_center/255. 
    op_data = op_data/255. 

    mix = bg_center * op_data 
    mix = np.round(( mix * 255. )).astype(np.uint8)
    bg_data[:, x:x+new_w,:] = mix 

    return x, new_w, bg_data

def outbbox(t_size, c1, c2, x, font):
    if len(c1)>len(c2):
        text = c1 
    else:
        text = c2 

    fontText = ImageFont.truetype(font, size=t_size, encoding="utf-8")
    tw,th = fontText.getsize(text)

    if tw> x-10 or th> 64-2:
        return True 
    else:
        return False

def get_content(op, c1, c2):
  
    sign = op.split('/')[-1].split('.')[0].split('-')[-1]
    content = ''.join([c1,sign,c2])
    return content
  


def create(bg_list, op_list):
    bg = random.choice(bg_list)
    op = random.choice(op_list)
    # 字体
    font = 'fonts/%02d.ttf' % (np.random.randint(0, 28))

    # 内容
    c1, c2 = random_content(), random_content()
    im_bg = Image.open(bg)
    im_op = Image.open(op)
    # imh, imw = im.size
    # print(c1,c2)

    # 先贴op
    x, new_w, im_s1 = paste_op(im_bg, im_op)
    im_s1 = Image.fromarray(im_s1)
    
    # 执行渲染
    draw = ImageDraw.Draw(im_s1)    
    t_size = np.random.randint(30,60) 
    fontText = ImageFont.truetype( font, t_size, encoding="utf-8")

    while outbbox(t_size, c1, c2, x, font):
        t_size -= 1
        fontsize = ImageFont.truetype( font, t_size, encoding="utf-8")
    
    fontText = ImageFont.truetype(font, t_size, encoding="utf-8")
  
    color = (0,0,0)
    h = 64
    
    tw,th = fontText.getsize(c1) 
    c1_left = int(x) - int(tw)
    c1_top = int(h//2) - int(th//2)
    draw.text((c1_left, c1_top), c1, color, font=fontText)

    # imgcat(np.array(im_s1))
    tw,th = fontText.getsize(c2) 
    c2_left = int(x+new_w)
    c2_top = int(h//2) - int(th//2)
    draw.text((c2_left, c2_top), c2, color, font=fontText)
    # imgcat(np.array(im_s1))
    # im.save(self.outname)
    content = get_content(op, c1, c2)
    return im_s1, content


if __name__ == "__main__":
    bg_dir = './datasets/backgrd'
    op_dir = './datasets/compare_op/'

    bg_list = get_list(bg_dir, 'jpg')
    op_list = get_list(op_dir, 'png')

    out_dir = './datasets/syh_002'
    out_img_dir = os.path.join(out_dir, 'images')
    makesure_path(out_dir)
    makesure_path(out_img_dir)
    label_path = os.path.join(out_dir, 'labels.txt')
    out_file = open(label_path,'w')

    NUM = int(sys.argv[1])
    for i in tqdm(range(NUM)):
        img, content = create(bg_list, op_list)    
        out_file.write('%05d.jpg,%s\n' % (i, content))
        outname = os.path.join(out_img_dir,'%05d.jpg' % i)
        img.save(outname)
    out_file.close()


    



