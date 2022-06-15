import sys, os 
import shutil

def get_list(path,type):
    list_out = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith(type)])
    return list_out

if __name__ == "__main__":
    root_dir = '../datasets/hand_written/'
    sub_list =['<','>','~','0',\
        '1','2','3','4','5','6','7','8','9','x','Y']
    for sub in sub_list:
        sub_dir = os.path.join(root_dir, sub)
        img_list = []
        img_list.extend(get_list(sub_dir,'jpg'))
        img_list.extend(get_list(sub_dir,'png'))
        print(img_list)
        sys.exit()
        for i, img_path in enumerate(img_list):
            new_name = os.path.join(sub_dir,'%03d.jpg' % i)
            os.rename(img_path, new_name)
    print('finish')
