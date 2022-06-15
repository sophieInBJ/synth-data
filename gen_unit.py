# 生成图
import sys, os
import random
from PIL import Image, ImageFont, ImageDraw
import numpy as np 
from imgcat import imgcat 
import cv2 
from tqdm import tqdm
from tqdm._tqdm import trange

def get_list(path,type):
    list_out = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith(type)])
    return list_out

def makesure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getUints(units_type):
    if units_type == 'time':
        return ['小时','分钟','分','秒','年','月','日','刻','半小时','毫秒']
    if units_type == 'long':
        return ['厘米','毫米','分米','米','公里','cm','m','dm']
    if units_type == 'weight':
        return ['千克','克','公斤','斤','吨','kg','t']
    if units_type == 'area':
        return ['平方米','平方分米','平方厘米','平方公里','公顷']
    if units_type == 'money':
        return ['元','角','分']
    if units_type == 'wan':
        return ['万','亿']
    if units_type == 'volume':
        return ['立方米','立方分米','立方厘米','毫升','升']

def random_num():   
    num1 = np.random.randint(0, 100)
    num2 = np.random.randint(0, 10)  
    return num1, num2

def random_unit():
    units_type = random.choice(['time','long','weight','area','money',\
        'wan','volume']) 
    uint_list = getUints(units_type)
    unit1 = random.choice(uint_list)
    unit2 = random.choice(uint_list)
    return unit1, unit2

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

def pre_paste(path):

    num_data = Image.open(path).convert("RGB")
    num_data = filter_op(num_data)
       
    return num_data

def resize_num3(num_datas):
    out = []
    for num_data in num_datas:
        op_h, op_w = num_data.shape[0], num_data.shape[1]
        new_w = int(64 * op_w/op_h)
        op_data = cv2.resize(num_data, (new_w, 64) ,interpolation=cv2.INTER_AREA)
        out.append(op_data)
    return out

def resize_paste(bg_data, num_data, end):
    bg = bg_data.copy()
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]
    # end = bg_w-1
   
    op_h, op_w = num_data.shape[0], num_data.shape[1]
    new_w = int(64 * op_w/op_h)
    op_data = cv2.resize(num_data, (new_w, 64) ,interpolation=cv2.INTER_AREA)
    
    bg_crop = bg[:, end:end+new_w,:]

    bg_crop = bg_crop/255. 
    op_data = op_data/255. 

    mix = bg_crop * op_data 
    mix = np.round(( mix * 255. )).astype(np.uint8)
    bg[:, end:end+new_w,:] = mix
    end = end+new_w
        # imgcat(bg)
    # sys.exit()
    return end, bg

def paste_num(bg, path_num, end):
    num_data = pre_paste(path_num)
    
    bg_data = np.array(bg)
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]
  
    w_use, bg_data = resize_paste(bg_data, num_data, end)
    
    return w_use, Image.fromarray(bg_data)


def outbbox(t_size, part1, part2, imshape, new_w, font):
    imh, imw = imshape
    
    fontText1 = ImageFont.truetype(font, size=t_size, encoding="utf-8")
    tw1,th1 = fontText1.getsize(part1)

    fontText2 = ImageFont.truetype(font, size=t_size, encoding="utf-8")
    tw2,th2 = fontText2.getsize(part2)

    total_w = new_w+tw1+tw2
    
    if total_w > imw-10 or th1> imh-2 or th2 > imh-2:
        return True 
    else:
        return False

def get_num2(num):
    # num =  np.random.randint(0, 10)
    c_list = char_list[num]
    return random.choice(c_list)

def compute_tsize(part1, part2, path_num2, imshape):
    bgh, bgw = imshape
    font = '../fonts/%02d.ttf' % (np.random.randint(0, 28))
    # 计算图片resize后的宽度
    num_data = cv2.imread(path_num2)
    op_h, op_w = num_data.shape[0], num_data.shape[1]
    new_w = int(bgh * op_w/op_h)

    t_size = np.random.randint(30,60) 
    while outbbox(t_size, part1, part2, imshape, new_w, font):
        t_size -= 1
    return t_size, font
        

def create(bg_list, char_list):
    bg = random.choice(bg_list)

    # 内容生成
    # num1 unit1 = (num2) unit2
    # 3km = (4)m 
    num1, num2 = random_num()
    unit1, unit2 = random_unit()
    path_num2 = get_num2(num2)
    part1 = '%s%s=(' % (num1, unit1)
    part2 = ')%s' % (unit2)

    # 打开背景图
    im_bg = Image.open(bg)
    bg_data = np.array(im_bg)
    bg_h, bg_w = bg_data.shape[0], bg_data.shape[1]

    # 提前计算好需要渲染的字体size 
    t_size, font = compute_tsize(part1, part2, path_num2, (bg_h, bg_w))
    fontText = ImageFont.truetype( font, t_size, encoding="utf-8")
  
    # 执行渲染
    draw = ImageDraw.Draw(im_bg)          
    color = (0,0,0)
    # 渲染第一段
    tw,th = fontText.getsize(part1) 
    ttop = 5
    tleft = 5
    # print(tw, th)
    draw.text((tleft, ttop), part1, color, font=fontText)

    end = tleft+tw
    # 贴图
    w_end, im_bg = paste_num(im_bg, path_num2, end)
    # 渲染第二段
    tw,th = fontText.getsize(part2) 
    ttop = 5
    tleft = w_end
    # print(tw, th)
    draw = ImageDraw.Draw(im_bg)   
    draw.text((tleft, ttop), part2, color, font=fontText)

    # imgcat(np.array(im_bg))
    # sys.exit()
    # im.save(self.outname)
    content = part1+str(num2)+part2
    
    return im_bg, content


if __name__ == "__main__":
    bg_dir = './datasets/backgrd'
    hd_dir = './datasets/hand_written/'
    
    # 图片路径
    bg_list = get_list(bg_dir, 'jpg')

    bg_list.remove('./datasets/backgrd/215.jpg')
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
    out_dir = '../datasets/syh_003'
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


    



