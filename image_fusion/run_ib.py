import fussion

import os
import time

import cv2
import random
import numpy as np
import json
from PIL import Image
import logging
logging.basicConfig(filename='./fall_log.txt',
                    level=logging.INFO, filemode='a', format='[%(asctime)s] [%(levelname)s] >>>  %(message)s',
                    datefmt='%Y-%m-%d %I:%M:%S')
                    
def cutout4seg(jsonPath, jpgDirPath, add_pic_path):
    jsonName = os.listdir(jsonPath)
    for num, json1 in enumerate(jsonName):
        print(num)
        path = jsonPath + json1
        j = 0
        with open(path, 'r') as f:
            load_dict = json.load(f)
            dic_data = load_dict["shapes"]
            imagePath = load_dict["imagePath"]
            img = cv2.imread(jpgDirPath + imagePath)
            for i in dic_data:
                pts = np.array(i["points"])
                pts = pts.astype(np.int64)
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect

                croped = img[y:y + h, x:x + w].copy()
                pts = pts - pts.min(axis=0)
                mask = np.zeros(croped.shape[:2], np.uint8)
                cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                dst = cv2.bitwise_and(croped, croped, mask=mask)
                bg = np.ones_like(croped, np.uint8) * 255
                cv2.bitwise_not(bg, bg, mask=mask)
                dst2 = bg + dst
                b_channel, g_channel, r_channel = cv2.split(dst2)  # 剥离jpg图像通道
                alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

                img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

                cv2.imwrite("{}/{}.png".format(add_pic_path, num), img_new)
                j += 1


class Transcolor():
    def __init__(self):
        self.color_map = {
            'white': (255, 255, 255, 0),
            'black': (0, 0, 0, 0),
        }

    def process(self, image_file, old_bk, new_bk, text_color):
        '''
        将图像特定颜色改为新颜色，前文改为设定颜色或者原始颜色
        '''
        img = Image.open(image_file).convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if self.is_around(item, old_bk):
                newData.append(new_bk)
            else:
                newData.append(text_color if text_color else item)
        img.putdata(newData)
        return img

    def transparent(self, image_file, bk_color='white', text_color=None):
        # 透明化
        bk = self.formulate(bk_color)
        text_color = self.formulate(text_color) if text_color else None
        return self.process(image_file, bk, (0, 0, 0, 0), text_color)

    def is_around(self, color1, color2):
        for i in range(3):
            if abs(color1[i] - color2[i]) > 30:
                return False
        return True

    def formulate(self, var):  # 格式检查
        if var in self.color_map.keys():
            return self.color_map[var]
        for n, i in enumerate(var):
            if i < 0 or i > 255 or n >= 4:
                print('Error:请输入white|black|phote_w|(220,220,220,0)RGBA形式')
                exit(1)
        return var

def change_color(img_add, v_num):
    """
    HSV颜色转换
    :param img_add:
    :return:
    """
    # 获取原图png石头的四个通道
    b1, g1, r1, m1 = cv2.split(img_add)
    # opencv的BGR转化为HSV
    # 其中H为色调取值范围：0-360
    # 其中S为饱和度取值范围：0-100
    # 其中V为明亮程度取值范围：0（黑）-100（白）
    img_hsv = cv2.cvtColor(img_add, cv2.COLOR_BGR2HSV)
    turn_green_hsv = img_hsv.copy()
    # # # 色调hue变化范围5——50
    # for i_h in range(1, 11):
    h_num = random.sample(range(0, 360), 1)[0]
    turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0] + h_num) % 180
    # turn_green_hsv[:, :, 2] = v_num * turn_green_hsv[:, :, 2]
    turn_green_img = cv2.cvtColor(turn_green_hsv, cv2.COLOR_HSV2BGR)
    b2, g2, r2 = cv2.split(turn_green_img)
    img_add = cv2.merge((b2, g2, r2, m1))
    return img_add

def judge_x1y1(json_path, old_w, old_h, add_w, add_h):
    if os.path.isfile(json_path):
        print("judge ---")
        with open(json_path, 'r', encoding='utf-8') as f:
            load_dict = json.load(f)
            dic_data = load_dict["shapes"]
            for i_seg in dic_data:
                pts = np.array(i_seg["points"])
                pts = pts.astype(np.int64)
                incides = random.sample(range(0, len(pts)), 3)
                (y1, x1), (y2, x2), (y3, x3) = pts[incides]

                # print((x1,y1),(x2,y2),(x3,y3))
                rnd1 = np.random.random(size=1)
                rnd2 = np.random.random(size=1)
                rnd2 = np.sqrt(rnd2)
                x1 = int(np.round(rnd2 * (rnd1 * x1 + (1 - rnd1) * x2) + (1 - rnd2) * x3))
                y1 = int(np.round(rnd2 * (rnd1 * y1 + (1 - rnd1) * y2) + (1 - rnd2) * y3))
    else:

        x1 = random.sample(range(int((old_w - add_w) / 2), old_w - add_w - 1), 1)[0]
        y1 = random.sample(range(0, old_h - add_h - 1), 1)[0]


    return x1, y1


# 旋转，R可控制图片放大缩小
def Rotate(image, angle=15, scale=0.9):
    # w = image.shape[1]
    # h = image.shape[0]
    # # rotate matrix
    # M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # # rotate
    # image = cv2.warpAffine(image, M, (w, h))
    # return image

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst


def change(old_pic_path, old_label_path, add_pic_path, new_pic_path, new_label_path):
    # yolo数据集中的cls_id
    cls_id = 0
    old_pic_path_list = os.listdir(old_pic_path)
    add_pic_path_list = os.listdir(add_pic_path)
    bad_i = 0
    # 遍历背景图
    for num, i in enumerate(old_pic_path_list):
        label_num = random.sample(range(1, 2), 1)[0]
        if len(add_pic_path_list)  < label_num * 1:
            add_pic_path_list = os.listdir(add_pic_path)
        logging.info(num)
        old_pic_all_path = os.path.join(old_pic_path, i)
        # 每张图贴n次，这样至少能保证label铁道铁轨上
        for j in range(1):
            # 读原始图片
            img_old = cv2.imread(old_pic_all_path)
            # 为原始jpg添加第四通道
            ones = np.ones((img_old.shape[0], img_old.shape[1])) * 255  # 创建一个一维数组
            img_alpha = np.dstack([img_old, ones])  # 添加第四通道
            img_alpha = img_alpha.astype('uint8')  # 上面处理时，像素值转为了 float，这里转回 uint8
            img_old = img_alpha
            # 随机选n个石头进行贴图
            add_stone_num = random.sample(range(1, len(add_pic_path_list)), label_num)
            add_stone_num.sort(reverse=True)
            with open("{}/{}_{}_{}_cover.txt".format(new_label_path, i.split(".")[0], num, j), "w") as f_write:
                # 把原来的标注也加进去
                if os.path.isfile("{}/{}.txt".format(old_label_path, i.split(".")[0])):
                    with open("{}/{}.txt".format(old_label_path, i.split(".")[0]), "r") as f_read:
                        f_write.write(f_read.read())
                for z in add_stone_num:
                    add_pic_all_path = os.path.join(add_pic_path, add_pic_path_list[z])
                    add_pic_path_list.pop(z)
                    img_add = cv2.imread(add_pic_all_path, cv2.IMREAD_UNCHANGED)
                    r_num = random.sample(range(0, 3), 1)[0]
                    old_w, old_h = img_old.shape[0], img_old.shape[1]
                    # 随机变换尺寸
                    r_w_num = random.sample(range(20, 65), 1)[0]
                    width = int(img_old.shape[1] * r_w_num / 1920)
                    height = int(width * img_add.shape[0] / img_add.shape[1])
                    dim = (width, height)
                    # resize image
                    img_add = cv2.resize(img_add, dim, interpolation=cv2.INTER_AREA)
                    add_w, add_h = img_add.shape[0], img_add.shape[1]
                    # 判断label图是否比背景图大
                    if add_w > old_w or add_h > old_h:
                        # print("-----------------------------", bad_i)
                        bad_i += 1
                        continue
                    json_path = os.path.join("old_json_label", "{}.json".format(i.split(".")[0]))
                    # print(json_path)
                    # 判断是否有label图片对应的json文件，再根据json文件中的外接矩形得x1, y1
                    x1, y1 = judge_x1y1(json_path, old_w, old_h, add_w, add_h)

                    # 控制要添加的区域
                    x1_limit = old_w - add_w - 1
                    y1_limit = old_h - add_h - 1
                    # 判断x1、y1是否超过图片限制
                    if x1 > x1_limit:
                        x1 = x1_limit
                    if y1 > y1_limit:
                        y1 = y1_limit
                    
                    source_file = add_pic_all_path
                    target_file = os.path.join(old_pic_path, i)
                    output_dir = './result/' + i.split(".")[0] + '_' + str(num) + '_' + str(j)
                    box = [x1, x1 + add_w, y1, y1 + add_h]
                    fussion.ib_box(source_file, target_file, output_dir, box)

                    # img_add1 和 img_add2 分别作为石头和原始图片被覆盖位置的像素比例，再分别乘以对应通道对应像素点然后加和，每个像素点范围在 0～255 之间，完成覆盖。
                    img_add1 = img_add[:, :, 3] / 255.0
                    img_add2 = 1 - img_add1
                    img_alpha2 = img_alpha.copy()
                    for c in range(4):
                        img_alpha2[x1: x1 + add_w, y1: y1 + add_h, c] = img_add2 * img_alpha[x1: x1 + add_w, y1: y1 + add_h, c] + img_add1 * img_add[:, :, c]
                    img_alpha = img_alpha2
                    # write yolo x,y,w,h
                    # 转化为标注格式的xywh，上面有关w和h的都为opecv格式，与实际相反
                    y = ((x1 + x1 + add_w) / 2 - 1) / old_w
                    x = ((y1 + y1 + add_h) / 2 - 1) / old_h
                    h = add_w / old_w
                    w = add_h / old_h
                    f_write.write(f"{cls_id} {x} {y} {w} {h}\n")
                    # jz_target_source_num
                    cv2.imwrite("{}/{}_{}_{}_cover.jpg".format(new_pic_path, i.split(".")[0], num, j), img_alpha2)


if __name__ == '__main__':
    old_pic_path = "old_pic"  # 原始图片路径
    add_pic_path = "add_pic"  # 需要加的label（石头）路径
    old_lable_path = "old_lable"  # 原始yolo数据集label路径
    new_pic_path = "new_pic"  # 最后生成的图片路径
    new_label_path = "new_label"  # 最后生成的yolo数据集label路径

    # 把石头覆盖到原始图中
    change(old_pic_path, old_lable_path, add_pic_path, new_pic_path, new_label_path)
