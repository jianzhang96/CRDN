import cv2, math
import random
import numpy as np
from scipy.spatial import ConvexHull

import torch
import torchvision.transforms as T

def pinch(img, degree=11):
    '''
    y = sqrt(x)
    :param img:
    :param degree: 1~32
    :return:
    '''
    if degree < 1: degree = 1
    if degree > 32: degree =32
    
    frame = cv2.imread(img) if isinstance(img, str) else img
    height, width, channels = frame.shape
    center_x = width / 2
    center_y = height / 2
    new_data = frame.copy()
    for i in range(width):
        for j in range(height):
            
            tx = i - center_x
            ty = j - center_y
            theta = math.atan2(ty, tx)
            radius = math.sqrt(tx ** 2 + ty ** 2) 

            radius = math.sqrt(radius) * degree 
            new_x = int(center_x + radius * math.cos(theta))
            new_y = int(center_y + radius * math.sin(theta))
            if new_x < 0:
                new_x = 0
            if new_x >= width:
                new_x = width - 1
            if new_y < 0:
                new_y = 0
            if new_y >= height:
                new_y = height - 1

            for channel in range(channels):
                new_data[j][i][channel] = frame[new_y][new_x][channel]
    return new_data

def pinch_range(img, range, degree):
    '''
    :param img:
    :param range: [x1,x2,y1,y2]
    :param degree:
    :return:
    '''
    
    x1,x2,y1,y2 = range
    img_r = img[x1:x2, y1:y2, :]
    dst_r = pinch(img_r, degree)
    img[x1:x2, y1:y2, :] = dst_r
    return img

def distort(img, degree, x, y, R=100):
    '''
    magnify
    :param img:
    :param x:
    :param y:
    :param degree: ratio
    :param R:
    :return:
    '''
    frame = cv2.imread(img) if isinstance(img, str) else img
    height, width, channels = frame.shape
    mid_x = x
    mid_y = y

    x1,x2 = x-R,x+R
    y1,y2 = y-R,y+R
    x1 = max(0,x1)
    y1 = max(0,y1)
    x2 = min(x2, width)
    y2 = min(y2, height)
    new_data = frame.copy()
    for i in range(x1,x2): 
        for j in range(y1,y2):
            tx = i - mid_x
            ty = j - mid_y
            theta = math.atan2(ty, tx)
            radius = math.sqrt(tx ** 2 + ty ** 2) 

            if radius <= R and radius > 1:
                k = math.sqrt(radius/R) * radius / R * degree
                new_x = int(math.cos(theta) * k + mid_x)
                new_y = int(math.sin(theta) * k + mid_y)

                if new_x < 0:
                    new_x = 0
                if new_x >= width:
                    new_x = width - 1
                if new_y < 0:
                    new_y = 0
                if new_y >= height:
                    new_y = height - 1

                for channel in range(channels):
                    new_data[j][i][channel] = frame[new_y][new_x][channel]

    return new_data


def sin_transform(img, rang, period=1, degree=3):
    '''
    :param img:
    :param degree:
    :return:
    '''
    frame = cv2.imread(img) if isinstance(img, str) else img
    height, width, channels = frame.shape
    new_data = np.zeros_like(frame)
    
    x1, x2, y1, y2 = rang
    if random.random() > 0.5:
        
        for j in range(y1,y2):
            temp = degree * math.sin(360 * j / width * math.pi/180 * period)   
            temp = degree + temp  
            for i in range(int(temp+0.5), int(height+temp-2*degree)):
                if x1 <= i <= x2:
                    x = int((i - temp) * height / (height - degree))
                    if x >= height:
                        x = height-1
                    if x < 0:
                        x = 0
                    for channel in range(channels):
                        new_data[i][j][channel] = frame[x][j][channel]
            for k in range(x1,x2+1):
                if new_data[k][j][0]==0:
                    new_data[k][j] = frame[x1][j]
    else:
        for j in range(x1,x2):
            temp = degree * math.sin(360 * j / height * math.pi/180 * period)   
            temp = degree + temp  
            for i in range(int(temp+0.5), int(width+temp-2*degree)):
                if y1 <= i <= y2:
                    x = int((i - temp) * width / (width - degree))
                    if x >= width:
                        x = width-1
                    if x < 0:
                        x = 0
                    for channel in range(channels):
                        new_data[j][i][channel] = frame[j][x][channel]
            for k in range(y1,y2+1):
                if new_data[j][k][0]==0:
                    new_data[j][k] = frame[j][y1]
    return new_data

def sin_trans_range(img, range, period=1, degree=3):
    '''
    :param img:
    :param range: [x1,x2,y1,y2]
    :param period:
    :param degree:
    :return:
    '''
    
    dst = sin_transform(img, range, period, degree)
    x1,x2,y1,y2 = range
    img[x1:x2, y1:y2, :] = dst[x1:x2, y1:y2, :]
    return img


def vortex(img, Para = 20):
    height, width, channels = img.shape
    center_x = width / 2
    center_y = height / 2
    
    dst = img.copy()
    for y in range(height):
        for x in range(width):
            tx = x - center_x
            ty = y - center_y
            theta = math.atan2(ty, tx)
            R = math.sqrt(tx ** 2 + ty ** 2) 
            delta = math.pi * Para / math.sqrt(R+1) 
            new_x = int(R * math.cos(theta + delta) + center_x)
            new_y = int(R * math.sin(theta + delta) + center_y)

            if new_x < 0:
                new_x = 0
            if new_x >= width:
                new_x = width - 1
            if new_y < 0:
                new_y = 0
            if new_y >= height:
                new_y = height - 1
            for channel in range(channels):
                dst[y][x][channel] = img[new_y][new_x][channel]
    return dst

def vortex_range(img, range ,Para = 5):
    '''
    :param img:
    :param range:
    :param Para: ratio
    :return:
    '''
    img0 = np.copy(img)
    x1,x2,y1,y2 = range
    img_r = img0[x1:x2, y1:y2, :]
    dst_r = vortex(img_r, Para)
    img0[x1:x2, y1:y2, :] = dst_r
    return img0

def frosted_glass(img, rang,offset,randp=False):
    '''
    :param img:
    :param offset: 10
    :return:
    '''
    height, width, deep = img.shape
    dst = np.copy(img)
    x1, x2, y1, y2 = rang
    
    randon_v = offset
    
    for n in range(y1, y2 - randon_v):
        for m in range(x1, x2 - randon_v):
            index = random.randint(1, randon_v)
            if randp and random.random()>0.6:
                (b, g, r) = (140/255,110/255,110/255)
            else:
                (b, g, r) = img[m+index, n+index]
            dst[m, n] = (b,g,r)
    
    for j in range(y2 - randon_v, y2 +1):
        for i in range(x1, x2+1):
            index = random.randint(1, randon_v)
            if randp and random.random()>0.6:
                (b, g, r) = (140/255,110/255,110/255)
            else:
                (b, g, r) = img[i - index, j - index]
            dst[i, j] = (b, g, r)
    for j in range(y1, y2+1):
        for i in range(x2 - randon_v, x2+1):
            index = random.randint(1, randon_v)
            if randp and random.random()>0.6:
                (b, g, r) = (140/255,110/255,110/255)
            else:
                (b, g, r) = img[i - index, j - index]
            dst[i, j] = (b, g, r)
    return dst

def frosted_glass_range(img, range, offset=10,randp=False):
    '''
    :param img:
    :param range:
    :param offset
    :return:
    '''
    img0 = np.copy(img)
    dst = frosted_glass(img0,range,offset,randp)
    x1,x2,y1,y2 = range
    img0[x1:x2, y1:y2, :] = dst[x1:x2, y1:y2, :]
    return img0

def perspective_transform(img, rang,mins=0.1,maxs=0.5):
    img0 = np.copy(img)
    
    cols,rows,_ = img.shape
    x1,x2,y1,y2 = rang
    w,h = abs(x2-x1),abs(y2-y1)
    pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

    tra = [(-w,-h),(w,-h),(w,h),(-w,h)]
    pts_t = []
    for i in range(4):
        wt,ht = tra[i]
        x = pts[i][0] + random.uniform(wt*0.1,wt*0.5)
        y = pts[i][1] + random.uniform(wt*0.1,ht*0.5)
        pts_t.append((x,y))

    p1 = np.float32(pts) 
    p2 = np.float32(pts_t) 
    M = cv2.getPerspectiveTransform(p1, p2)
    dst = cv2.warpPerspective(img0, M, (cols, rows)) 
    return dst


def wind_effect(image,num=12,num1=40):
    height, width, channels = image.shape

    dst = image.copy()
    for y in range(height):
        dens = int(random.uniform(num*3//4, num))
        for i in range(dens): 
            newX = int(random.uniform(int(i * width / num), int((i + 1) * width / num)))
            newY = y

            if newX < 0: newX = 0
            if newX > width - 1: newX = width-1
            numl = int(random.uniform(num1//2, num1))
            for j in range(numl):
                tmpX = newX - j 
                if tmpX < 0: tmpX=0
                if tmpX > width - 1: tmpX=width-1
                dst[y][tmpX][0] = image[newY][newX][0]
                dst[y][tmpX][1] = image[newY][newX][1]
                dst[y][tmpX][2] = image[newY][newX][2]
        
    
    angle = int(random.uniform(2,9))
    if angle<5: scale = 1.3
    else: scale = 1.6
    dst = rotate_bound_black_bg(dst,angle,scale)
    return dst

def wind_effect_range(img,rang):
    img0 = np.copy(img)
    x1,x2,y1,y2 = rang
    if x1>40:
        img_r = img0[x1-40:x2, y1:y2, :]
    else:
        img_r = img0[:x2, y1:y2, :]
    dst_r = wind_effect(img_r)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  
    if x1>40:
        dst = cv2.filter2D(dst_r[40:,:,:], -1, kernel=kernel)
    else:
        dst = cv2.filter2D(dst_r[x1:, :, :], -1, kernel=kernel)
    img0[x1:x2, y1:y2, :] = dst
    return img0


def rotate_bound_black_bg(image, angle,scale):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    rotated = cv2.warpAffine(image,M,(w,h))
    return rotated

def color_jitter(image,rang,br=0.1,hue=0.1):
    x1,x2,y1,y2 = rang
    img = np.copy(image)
    img_t = torch.from_numpy(img[x1:x2, y1:y2, :])
    jitter = T.ColorJitter(brightness=br, hue=hue,contrast=0.1, saturation=0.1)
    jitted_img = jitter(img_t.permute(2,0,1))
    jitted_img = jitted_img.permute(1,2,0).numpy()
    img[x1:x2, y1:y2, :] = jitted_img
    return img

def grayscale_range(image,rang):
    x1,x2,y1,y2 = rang
    img = np.copy(image)
    img_t = torch.from_numpy(img[x1:x2, y1:y2, :])
    gray_img = T.Grayscale()(img_t.permute(2,0,1))
    gray_img = gray_img.permute(1,2,0).numpy()
    img[x1:x2, y1:y2, :] = gray_img
    return img

def cutout_range(image,rang,val=0.65):
    img = np.copy(image)
    x1,x2,y1,y2 = rang
    img[x1:x2,y1:y2] = val
    return img

def get_image_binary(image):
    
    img_o = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_o,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = binary / 255.0
    return np.expand_dims(binary,2)

def get_image_contour(image):
    img_o = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_o,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours,key=lambda x:len(x))
    return cnt


def sample_contours(image,cnt):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    pixelpoints = cv2.findNonZero(mask)
    l = len(pixelpoints)
    
    if random.random() > 0.5: w,h = 10,20
    else: w, h = 20,10

    while True:
        idx = np.random.randint(0, l, 1)
        pts = pixelpoints[idx, :, :]
        x,y = pts.flatten()
        wid,hei = image.shape[:2]
        res = [y,y+h,x,x+w]
        p1 = cv2.pointPolygonTest(cnt,(int(x),int(y+w)),False)
        p2 = cv2.pointPolygonTest(cnt,(int(x+h),int(y)),False)
        p3 = cv2.pointPolygonTest(cnt,(int(x+h),int(y+w)),False)
        if p1>0 and p2>0 and p3>0 and y+h < hei and x+w < wid:
            break
    return res

def arch_augment_fast(image0):
    '''
    sag
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    center_x, center_y = int(width / 2), int(height / 2)
    mask = np.zeros_like(image0) 

    range4 = []
    addp = [(0,0),(center_x,0),(0,center_y),(center_x,center_y)]
    whs = np.random.randint(30//4, 250//4, 8)

    for i in range(4):
        w, h = whs[i:i+2]
        x = np.random.randint(0, center_x - w)
        y = np.random.randint(0, center_y - h)
        offx, offy = addp[i][0], addp[i][1]
        range4.append([x+offx,x+w+offx,y+offy,y+h+offy])
    range_id = [0,1,2,3]
    random.shuffle(range_id)
    
    x1,x2,y1,y2 = range4[range_id[1]]
    x = (x1 + x2)//2
    y = (y1 + y2)//2
    R = np.random.randint(10//4, 120//4)
    degree = get_degreebyR(R)
    image = distort(image0,degree,x,y,R)
    cv2.circle(mask, (x, y), R, (1, 1, 1), -1)

    
    degree = np.random.randint(5, 8)
    if degree >6: degree = np.random.randint(5, 8)
    rang = range4[range_id[0]]
    polyp = getPolygonArea(image,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = pinch_range(image,rang,degree) 
    image = image * mask + image0 * (1 - mask)
    
    degree = np.random.randint(16, 31)
    if degree > 22: degree = np.random.randint(16, 31)
    rang = range4[range_id[2]]
    polyp= getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree) 
    image = image * mask + image0 * (1 - mask)

    para = np.random.randint(1, 8)
    rang = range4[range_id[3]]
    polyp = getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = vortex_range(image,rang,para) 
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)


def arch_augment_grid(image0):
    '''
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0) 

    w, h = np.random.randint(30 // 4, 250 // 5, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+w,y,y+h)

    para = np.random.randint(1, 8)
    polyp = getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = vortex_range(image0,rang,para) 
    image = image * mask + image0 * (1 - mask)

    degree = np.random.randint(16, 31)
    w, h = np.random.randint(30 // 4, 250 // 5, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+w,y,y+h)

    polyp= getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree) 
    image = image * mask + image0 * (1 - mask)

    
    degree = np.random.randint(5, 8)
    w, h = np.random.randint(30 // 4, 250 // 5, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+w,y,y+h)

    polyp = getPolygonArea(image,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

    image = pinch_range(image,rang,degree) 
    image = image * mask + image0 * (1 - mask)

    
    w, h = np.random.randint(30 // 4, 250 // 5, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+w,y,y+h)

    polyp = getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    binary = get_image_binary(image)
    mask = mask * binary
    image = cutout_range(image, rang)
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)

def arch_augment_screw(image0):
    '''
    four non-linear transformation
    :param image:
    :return: anomaly image, mask, whether anomaly
    '''
    mask = np.zeros_like(image0)

    cnt = get_image_contour(image0)
    
    para = np.random.randint(1, 8)
    rang = sample_contours(image0,cnt)
    polyp = getPolygonArea(image0, rang)

    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = vortex_range(image0,rang,para) 
    image = color_jitter(image,rang,0.3,0.3)
    image = image * mask + image0 * (1 - mask)

    rang = sample_contours(image0,cnt)

    y1,y2,x1,x2 = rang
    x = (x1 + x2)//2
    y = (y1 + y2)//2
    R = np.random.randint(3, 8)
    degree = get_degreebyR(R)
    
    image = distort(image,degree,x,y,R)
    image = color_jitter(image,rang,0.3,0.3)
    
    cv2.circle(mask, (x, y), R, (1, 1, 1), -1)

    degree = np.random.randint(5, 8)
    rang = sample_contours(image0,cnt)

    polyp = getPolygonArea(image0,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

    image = pinch_range(image,rang,degree) 
    image = color_jitter(image,rang,0.3,0.3)
    image = image * mask + image0 * (1 - mask)
    
    degree = np.random.randint(16, 31)
    rang = sample_contours(image0,cnt)

    polyp= getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree)
    image = color_jitter(image,rang,0.3,0.3)
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([1.0],dtype=np.float32)

def arch_augment_toothb(image0):
    '''
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0)

    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(80, 180 - h)
    rang = (x,x+w,y,y+h)

    polyp = getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = perspective_transform(image0,rang)
    image = image * mask + image0 * (1 - mask)
    
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(80, 180 - h)
    rang = (x,x+w,y,y+h)

    x1,x2,y1,y2 = rang
    x = (x1 + x2)//2
    y = (y1 + y2)//2
    R = np.random.randint(16//4, 120//4)
    degree = get_degreebyR(R)
    
    image = distort(image,degree,x,y,R)
    cv2.circle(mask, (x, y), R, (1, 1, 1), -1)

    degree = np.random.randint(5, 8)
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(80, 180 - h)
    rang = (x, x + w, y, y + h)

    polyp = getPolygonArea(image,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

    image = pinch_range(image,rang,degree) 
    image = image * mask + image0 * (1 - mask)

    degree = np.random.randint(16, 31)
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(80, 180 - h)
    rang = (x, x + w, y, y + h)

    polyp= getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree)
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([1.0],dtype=np.float32)

def arch_augment_leather(image0):
    '''
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0) 

    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+w,y,y+h)

    polyp = getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = perspective_transform(image0,rang)
    image = image * mask + image0 * (1 - mask)

    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+w,y,y+h)

    x1,x2,y1,y2 = rang
    x = (x1 + x2)//2
    y = (y1 + y2)//2
    R = np.random.randint(16//4, 120//4)
    degree = get_degreebyR(R)
    
    image = distort(image,degree+15,x,y,R)
    
    cv2.circle(mask, (x, y), R, (1, 1, 1), -1)
    
    degree = np.random.randint(5, 8)
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x, x + w, y, y + h)

    polyp = getPolygonArea(image,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

    image = pinch_range(image,rang,degree) 
    image = image * mask + image0 * (1 - mask)
    
    degree = np.random.randint(16, 31)
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x, x + w, y, y + h)

    polyp= getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree) 
    image = image * mask + image0 * (1 - mask)

    
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+h,y,y+w)

    para = np.random.randint(1, 8)
    polyp = getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = vortex_range(image,rang,para) 
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)

def arch_augment_tile(image0):
    '''
    frosted glass
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0) 

    w, h = np.random.randint(30 , 250 // 2, 2)
    x = np.random.randint(0, width - w)
    y = np.random.randint(0, height - h)
    rang = (x,x+w,y,y+h)

    para = np.random.randint(12, 16)
    polyp = getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = frosted_glass_range(image0,rang,para)
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)

def arch_augment_capsule(image0):
    '''
    :param image:
    :return:
    '''
    mask = np.zeros_like(image0)

    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    y = np.random.randint(18, max(241 - w,18))
    x = np.random.randint(92, max(164 - h,92))
    rang = (x,x+h,y,y+w)

    para = np.random.randint(1, 8)
    polyp = getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = vortex_range(image0,rang,para) 
    image = image * mask + image0 * (1 - mask)
    
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    y = np.random.randint(18, max(241 - w,18))
    x = np.random.randint(92, max(164 - h,92))
    rang = (x,x+h,y,y+w)

    x1,x2,y1,y2 = rang
    x = (x1 + x2)//2
    y = (y1 + y2)//2
    R = np.random.randint(16//4, 120//4)
    degree = get_degreebyR(R)
    
    image = distort(image,degree,y,x,R)
    
    cv2.circle(mask, (y, x), R, (1, 1, 1), -1)

    degree = np.random.randint(5, 8)
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    y = np.random.randint(18, max(241 - w,18))
    x = np.random.randint(92, max(164 - h,92))
    rang = (x,x+h,y,y+w)

    polyp = getPolygonArea(image,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

    image = pinch_range(image,rang,degree) 
    image = image * mask + image0 * (1 - mask)
    
    degree = np.random.randint(16, 31)
    w, h = np.random.randint(30 // 4, 250 // 4, 2)
    y = np.random.randint(18, 241 - w)
    x = np.random.randint(92, 164 - h)
    rang = (x,x+h,y,y+w)

    polyp= getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree) 
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)

def arch_augment_zipper(image0):
    '''
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0)
    
    w, h = np.random.randint(30 // 4, 250 // 6, 2)
    y = np.random.randint(112, max(145 - w, 113))
    x = np.random.randint(0, height - h)
    rang = (x,x+h,y,y+w)

    para = np.random.randint(1, 8)
    polyp = getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = vortex_range(image0,rang,para) 
    image = image * mask + image0 * (1 - mask)

    w, h = np.random.randint(30 // 4, 250 // 6, 2)
    y = np.random.randint(112, max(145 - w, 113))
    x = np.random.randint(0, height - h)
    rang = (x,x+h,y,y+w)

    x1,x2,y1,y2 = rang
    x = (x1 + x2)//2
    y = (y1 + y2)//2
    R = np.random.randint(16//4, 120//4)
    degree = get_degreebyR(R)
    
    image = distort(image,degree,y,x,R)
    cv2.circle(mask, (y, x), R, (1, 1, 1), -1)
    
    degree = np.random.randint(5, 8)
    w, h = np.random.randint(30 // 4, 250 // 6, 2)
    y = np.random.randint(112, max(145 - w, 113))
    x = np.random.randint(0, height - h)
    rang = (x,x+h,y,y+w)

    polyp = getPolygonArea(image,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

    image = pinch_range(image,rang,degree) 
    image = image * mask + image0 * (1 - mask)
    
    degree = np.random.randint(16, 31)
    w, h = np.random.randint(30 // 4, 250 // 6, 2)
    y = np.random.randint(112, max(145 - w, 113))
    x = np.random.randint(0, height - h)
    rang = (x,x+h,y,y+w)

    polyp= getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree)
    image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)

def get_degreebyR(R):
    '''
    :param R:
    :return:
    '''
    if 10//4<=R<20//4:
        degree = np.random.randint(R+30//3, R+40//3)
    elif 20//4<=R<50//4:
        degree = np.random.randint(R+20//2, R+30//2)
    elif 50//4<=R<70//4:
        degree = np.random.randint(R+15//2, R+25//2)
    elif 70//4<=R<100//4:
        degree = np.random.randint(R+10//3, R+20//2)
    elif 100//4<=R<120//4:
        degree = np.random.randint(R+3, R+13)
    return degree


def getPolygonArea(img,rang,sam_nums=20):
    '''
    :param rang:
    :param sam_nums:
    :return:
    '''
    x1,x2,y1,y2 = rang

    points_x = np.random.randint(low=x1,high=x2,size=(sam_nums,), dtype=np.int32)
    points_y = np.random.randint(low=y1,high=y2,size=(sam_nums,), dtype=np.int32)
    points = np.vstack((points_y,points_x)).transpose(1,0)

    hull = ConvexHull(points)

    pts = points[hull.vertices]
    return pts

###
def arch_augment_btad01(image0):
    '''
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0)

    r = 128
    #
    if random.random() > 0.2:
        w, h = np.random.randint(20, 60, 2)
    else: #
        w = int(np.random.randint(3, 10, 1))
        h = int(np.random.randint(50, 70, 1))
    while True:
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+w,y,y+h)
        c_x, c_y = x+w//2, y+h//2
        dist = ((c_x - 128)**2 + (c_y - 128)**2)**0.5
        if dist + w/2 < r and 48 < dist - w/2:
            break

    polyp = getPolygonArea(image0, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = wind_effect_range(image0,rang)
    image = image * mask + image0 * (1 - mask)

    #
    degree = np.random.randint(5, 8)
    while True:
        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+w,y,y+h)
        c_x, c_y = x+w//2, y+h//2
        dist = ((c_x - 128)**2 + (c_y - 128)**2)**0.5
        if dist + w/2 < r and 48 < dist - w/2:
            break

    polyp = getPolygonArea(image,rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

    image = pinch_range(image,rang,degree,True)
    image = image * mask + image0 * (1 - mask)

    #
    degree = np.random.randint(16, 31)
    while True:
        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+w,y,y+h)
        c_x, c_y = x+w//2, y+h//2
        dist = ((c_x - 128)**2 + (c_y - 128)**2)**0.5
        if dist + w/2 < r and 48 < dist - w/2:
            break

    polyp= getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = sin_trans_range(image,rang,2,degree)
    image = image * mask + image0 * (1 - mask)

    #
    while True:
        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+w,y,y+h)
        c_x, c_y = x+w//2, y+h//2
        dist = ((c_x - 128)**2 + (c_y - 128)**2)**0.5
        if dist + w/2 < r and 48 < dist - w/2:
            break

    para = np.random.randint(1, 8)
    polyp = getPolygonArea(image, rang)
    mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
    image = vortex_range(image,rang,para) # 1~8
    image = image * mask + image0 * (1 - mask)

    if random.random() > 0.5:
        if random.random() > 0.5:
            image = cv2.GaussianBlur(image,(3,3),0)
        else:
            image = cv2.blur(image,(3,3))


    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)

def arch_augment_btad02(image0):
    '''
    :param image:
    :return:
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0)
    if random.random() > 0.2:
        if random.random() > 0.5:
            w, h = np.random.randint(70, 80, 2)
        else:
            w = int(np.random.randint(3, 6, 1))
            h = int(np.random.randint(50, 70, 1))
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+w,y,y+h)

        polyp = getPolygonArea(image0, rang)
        mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
        image = wind_effect_range(image0,rang)
        image = image * mask + image0 * (1 - mask)
    else:
        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+w,y,y+h)

        polyp = getPolygonArea(image0, rang)
        mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
        image = perspective_transform(image0,rang)
        image = image * mask + image0 * (1 - mask)

        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+w,y,y+h)

        x1,x2,y1,y2 = rang
        x = (x1 + x2)//2
        y = (y1 + y2)//2
        R = np.random.randint(16//4, 120//4)
        degree = get_degreebyR(R)
        image = distort(image,degree+15,x,y,R)
        cv2.circle(mask, (x, y), R, (1, 1, 1), -1)

        #
        degree = np.random.randint(5, 8)
        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x, x + w, y, y + h)

        polyp = getPolygonArea(image,rang)
        mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

        image = pinch_range(image,rang,degree) # degree:7~12
        image = image * mask + image0 * (1 - mask)

        #
        degree = np.random.randint(16, 31)
        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x, x + w, y, y + h)

        polyp= getPolygonArea(image, rang)
        mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
        image = sin_trans_range(image,rang,2,degree) # period:3 degree:30~120
        image = image * mask + image0 * (1 - mask)

        #
        w, h = np.random.randint(30 // 4, 250 // 4, 2)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        rang = (x,x+h,y,y+w)

        para = np.random.randint(1, 8)
        polyp = getPolygonArea(image, rang)
        mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
        image = vortex_range(image,rang,para) # 1~8
        image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32)

def cutout_gradual_color(image,center,axesl,cnt,val=(84,170,198)):
    img = np.copy(image)
    c = (val[2]/255,val[1]/255,val[0]/255)
    x0, y0, w0, h0 = cv2.boundingRect(cnt)
    x1,x2 = x0,x0+w0
    y1,y2 = y0,y0+h0

    c_x, c_y = center
    # x1, x2 = c_x - cr, c_x + cr
    # y1, y2 = c_y - cr, c_y + cr
    # img[x1:x2,y1:y2] = c
    R = axesl[0]
    for y in range(y1, y2):
        for x in range(x1, x2):
            # r = ((x-c_x)**2 + (y-c_y)**2)**0.5
            r = cv2.pointPolygonTest(cnt, (x, y), True)
            if r < 0:
                img[y, x] = (0, 0, 0)
            # if r > R:
            #     img[y,x] = (0,0,0)
            else:
                c0 = (1-r/R)* c[0] + r/R*1.5
                c1 = (1-r/R)* c[1] + r/R*1.5
                c2 = (1-r/R)* c[2] + r/R*1.5
                c0 = c0 if c0 <= 1 else 1
                c1 = c1 if c1 <= 1 else 1
                c2 = c2 if c2 <= 1 else 1
                img[y,x] = (c0,c1,c2)
    return img

def cutout_gradual_color_circle(image,center,cr,val=(84,170,198)):
    img = np.copy(image)
    c = (val[2]/255,val[1]/255,val[0]/255)

    c_x, c_y = center
    x1, x2 = c_x - cr, c_x + cr
    y1, y2 = c_y - cr, c_y + cr
    R = cr
    for y in range(y1, y2):
        for x in range(x1, x2):
            r = ((x-c_x)**2 + (y-c_y)**2)**0.5
            if r > R:
                img[y,x] = (0,0,0)
            else:
                c0 = (1-r/R) + r/R * c[0]
                c1 = (1-r/R) + r/R * c[1]
                c2 = (1-r/R) + r/R * c[2]
                img[y,x] = (c0,c1,c2)

    return img

def arch_augment_btad03(image0):
    '''
    '''
    height, width, ch = image0.shape
    mask = np.zeros_like(image0)
    r = 123

    if random.random() > 0.05:
        while True:
            sl = np.random.randint(7, 18)
            ll = np.random.randint(10, 50)
            angle = np.random.randint(0, 360)
            cx = np.random.randint(ll, width - ll)
            cy = np.random.randint(ll, height - ll)
            dist = ((cx - 128)**2 + (cy - 128)**2)**0.5
            if dist + ll < r and 30 < dist - ll:
                axesl = (ll,sl)
                cv2.ellipse(mask,(cx, cy),axesl, angle,0,360, (1, 1, 1), -1)
                contours, hie = cv2.findContours((mask[:,:,0]*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt = contours[0]
                image = cutout_gradual_color(image0, (cx, cy), axesl,cnt)
                image = image * mask + image0 * (1 - mask)
                break

        while True:
            cr = np.random.randint(7, 30)
            cx = np.random.randint(cr, width - cr)
            cy = np.random.randint(cr, height - cr)
            dist = ((cx - 128)**2 + (cy - 128)**2)**0.5
            if dist + ll < r and 30 < dist - ll:
                cv2.circle(mask, (cx, cy), cr, (1, 1, 1), -1)
                image = cutout_gradual_color_circle(image, (cx, cy),cr)
                image = image * mask + image0 * (1 - mask)
                break
    else:
        # #挤压
        degree = np.random.randint(5, 8)
        while True:
            w, h = np.random.randint(30 // 4, 250 // 4, 2)
            x = np.random.randint(0, width - w)
            y = np.random.randint(0, height - h)
            rang = (x,x+w,y,y+h)
            c_x, c_y = x+w//2, y+h//2
            dist = ((c_x - 128)**2 + (c_y - 128)**2)**0.5
            if dist + w/2 < r and 30 < dist - w/2:
                break
        #
        polyp = getPolygonArea(image0,rang)
        mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))

        image = pinch_range(image0,rang,degree) # degree:7~12
        image = image * mask + image0 * (1 - mask)
        #

        while True:
            w, h = np.random.randint(30 // 4, 250 // 4, 2)
            x = np.random.randint(0, width - w)
            y = np.random.randint(0, height - h)
            rang = (x,x+w,y,y+h)
            c_x, c_y = x+w//2, y+h//2
            dist = ((c_x - 128)**2 + (c_y - 128)**2)**0.5
            if dist + w/2 < r and 30 < dist - w/2:
                break

        para = np.random.randint(1, 8)
        polyp = getPolygonArea(image, rang)
        mask = cv2.fillPoly(mask, [polyp], color=(1, 1, 1))
        image = vortex_range(image,rang,para) #
        image = image * mask + image0 * (1 - mask)

    mask = mask[:,:,0].astype(np.float32)
    mask = np.expand_dims(mask, axis=2)
    return image, mask, np.array([0.0],dtype=np.float32), image0
###