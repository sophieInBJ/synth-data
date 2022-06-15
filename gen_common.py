# 生成图
import sys, os
import random
from PIL import Image, ImageFont, ImageDraw
import numpy as np 
import cv2 
from tqdm import tqdm


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
    threshold = 160 
    op_data = np.array(op)
    # imgcat(op_data)
    mask = np.where(op_data>threshold)
    nomask = np.where(op_data<=threshold)
    op_data[mask] = 255
    k = 255.0/threshold
    op_data[nomask] = op_data[nomask]*k
    op_data = np.uint8(op_data)

    return op_data

def pre_paste(path_nums):
    outs = []
    for path in path_nums:
        # print(path)
        num_data = Image.open(path).convert("RGB")
        num_data = filter_op(num_data)
        outs.append(num_data)
    return outs

def resize_num3(num_datas):
    out = []
    for num_data in num_datas:
        op_h, op_w = num_data.shape[0], num_data.shape[1]
        new_w = int(64 * op_w/op_h)
        op_data = cv2.resize(num_data, (new_w, 64) ,interpolation=cv2.INTER_AREA)
        out.append(op_data)
    return out

def resize_paste(bg_data, num_datas):
    bg = bg_data.copy()
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]
    end = bg_w-1
    for num_data in num_datas:
        op_h, op_w = num_data.shape[0], num_data.shape[1]
        new_w = int(64 * op_w/op_h)
        op_data = cv2.resize(num_data, (new_w, 64) ,interpolation=cv2.INTER_AREA)
        
        bg_crop = bg[:, end-new_w:end,:]

        bg_crop = bg_crop/255. 
        op_data = op_data/255. 

        mix = bg_crop * op_data 
        mix = np.round(( mix * 255. )).astype(np.uint8)
        bg[:, end-new_w:end,:] = mix
        end = end-new_w+10
 
    return end, bg

def paste_num3(bg, path_nums):
    num_datas = pre_paste(path_nums)
    
    bg_data = np.array(bg)
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]
  
    w_use, bg_data = resize_paste(bg_data, num_datas)
    
    return w_use, bg_data


def outbbox(t_size, text, w, font):
  
    fontText = ImageFont.truetype(font, size=t_size, encoding="utf-8")
    tw,th = fontText.getsize(text)

    if tw> w-10 or th> 64-2:
        return True 
    else:
        return False

def get_content(op, c1, c2):
  
    sign = op.split('/')[-1].split('.')[0].split('-')[-1]
    content = ''.join([c1,sign,c2])
    return content

def do_random_char(char_list):
    num =  np.random.randint(0, 10)
    c_list = char_list[num]
    return [ random.choice(c_list), str(num)]

def random_num3(char_list):
    str_list = []
    outs = []
    nlen = random.choice(['1','2','3'])
    # first
    pic_path, str_ = do_random_char(char_list)
    outs.append(pic_path)
    str_list.append(str_)

    if nlen == '2':
        pic_path, str_ = do_random_char(char_list)
        outs.append(pic_path)
        str_list.append(str_)
       
    if nlen == '3':
        pic_path, str_ = do_random_char(char_list)
        outs.append(pic_path)
        str_list.append(str_)

        pic_path, str_ = do_random_char(char_list)
        outs.append(pic_path)
        str_list.append(str_)
    str_list = reversed(str_list)

    return outs, ''.join(str_list)

def create(bg_list, char_list):
    bg = random.choice(bg_list)
    # op = random.choice(op_list)
    # 字体
    font = './fonts/%02d.ttf' % (np.random.randint(0, 28))

    # 内容 
    # num1 op num2 = num3
    left = random_AB()+'='
    path_nums, str_num3 = random_num3(char_list)
     
    # 打开背景图
    im_bg = Image.open(bg)
    bg_data = np.array(im_bg)
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]


    # 先贴结果
    w_end, im_s1 = paste_num3(im_bg, path_nums)
    im_s1 = Image.fromarray(im_s1)
  
    # 执行渲染
    draw = ImageDraw.Draw(im_s1)    
    t_size = np.random.randint(30,60) 
    fontText = ImageFont.truetype( font, t_size, encoding="utf-8")
    content = left
    while outbbox(t_size, content, w_end, font):
        t_size -= 1
        fontText = ImageFont.truetype( font, t_size, encoding="utf-8")
      
    color = (0,0,0)
    h = 64
    
    tw,th = fontText.getsize(content) 
    ttop = 5
    tleft = w_end-tw
    draw.text((tleft, ttop), content, color, font=fontText)


    content = left + str_num3
    
    return im_s1, content


if __name__ == "__main__":
    bg_dir = './datasets/backgrd'
    hd_dir = './datasets/hand_written/'
    
    # 图片路径
    bg_list = get_list(bg_dir, 'jpg')
    char0_list = get_list(hd_dir+'0', 'jpg')
    char1_list = get_list(hd_dir+'1', 'jpg')
    char2_list = get_list(hd_dir+'2', 'jpg')
    char3_list = get_list(hd_dir+'3', 'jpg')
    char4_list = get_list(hd_dir+'4', 'jpg')
    char5_list = get_list(hd_dir+'5', 'jpg')
    char6_list = get_list(hd_dir+'6', 'jpg')
    char7_list = get_list(hd_dir+'7', 'jpg')
    char8_list = get_list(hd_dir+'8', 'jpg')
    char9_list = get_list(hd_dir+'9', 'jpg')

    char_list = [char0_list, char1_list, char2_list,\
        char3_list, char4_list, char5_list, char6_list,\
        char7_list, char8_list, char9_list]
    
    # 输出路径
    out_dir = './datasets/syh_001'
    out_img_dir = os.path.join(out_dir, 'images')
    makesure_path(out_dir)
    makesure_path(out_img_dir)
    label_path = os.path.join(out_dir, 'labels.txt')
    out_file = open(label_path,'w')

    NUM = int(sys.argv[1])

    for i in tqdm(range(NUM)):
        img, content = create(bg_list, char_list)    
        out_file.write('%05d.jpg,%s\n' % (i, content))
        outname = os.path.join(out_img_dir,'%05d.jpg' % i)
        img.save(outname)
  
    out_file.close()
    print('finish')


    



