import numpy as np
import cv2
from time import time


# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if (rect[0] + rect[2] > limit[0] + limit[2]):
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3] > limit[1] + limit[3]):
        rect[3] = limit[1] + limit[3] - rect[1]
    if (rect[0] < limit[0]):
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if (rect[1] < limit[1]):
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if (rect[2] < 0):
        rect[2] = 0
    if (rect[3] < 0):
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


# Simple tracker
class BaseTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        self._interp_factor = 0.075  # model updating rate
        self._tmpl_sz = np.array([50, 50])  # the fixed model size
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [left_up_x,left_up_y,width,height]
        self._tmpl = None  # our model
        # self._scale_pool = [0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015]

    def getFeatures(self, image, roi, needed_size):
        roi = list(map(int, roi))  # ensure that everything is int

        # roi = list(roi)

        z = subwindow(image, roi, cv2.BORDER_REPLICATE)  # sample a image patch
        if z.shape[1] != needed_size[0] or z.shape[0] != needed_size[1]:
            z = cv2.resize(z, tuple(needed_size))  # resize to template size

        if z.ndim == 3 and z.shape[2] == 3:
            FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        elif z.ndim == 2:
            FeaturesMap = z  # (size_patch[0], size_patch[1]) #np.int8  #0~255
        FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5

        return FeaturesMap

    def measure(self, a, b):
        """
        This is the measure function you need to code
        you should use self.measure in self.track to simplify your code
        """
        los = (np.sum((a - b) ** 2))
        return los

    def track(self, search_region, img): # search roi ; one frame img
        """
        this is the main place you need to code
        please check ln.117 to see what's the input of this function
        """

        minmse = 1e8
        retx, rety = 0,0

        listi = self.getSearchIdx(int(search_region[2]/2))  # 获取search region的采样，中间密，两边疏
        listj = self.getSearchIdx(int(search_region[3]/2))

        # for i in range(0, int(search_region[2]/2), 2):
        #     for j in range(0, int(search_region[3] / 2), 2):
        for i in listi:
            for j in listj:
                new = self.getFeatures(img, [search_region[0]+i,search_region[1]+j,self._roi[2], self._roi[3]], self._tmpl_sz)
                mse = self.measure(self._tmpl, new)
                # print("i", i)
                # print('mse', mse)
                if mse < minmse:
                    minmse = mse
                    retx = i
                    rety = j

        # print("ret", retx,'/', int(search_region[2]/2) , rety, '/', int(search_region[3]/2))
        return retx, rety, 1.0025  # 返回目标区域的左上角坐标（相对于search region的），超参调了一个1.0025的roi scale增长率


    def update_model(self, x, train_interp_factor):
        """
        This is the model update function you need to code
        """
        self._tmpl = self._tmpl * (1 - train_interp_factor) + train_interp_factor * x


    def init(self, roi, image):
        self._roi = list(map(int, roi))

        assert (roi[2] > 0 and roi[3] > 0)
        self._tmpl = self.getFeatures(image, self._roi, self._tmpl_sz)
        # print(self._tmpl)

    def update(self, image):
        # self._roi = list(self._roi)
        # print(self._roi)
        # some check boundary here
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[2] + 1
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 2
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 2
        # center position of our target
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.
        # we double the searching region compared to the selected region
        search_rect = [cx - self._roi[2], cy - self._roi[3], self._roi[2] * 2, self._roi[3] * 2]
        # the delta in search region
        loc_pos = self.track(search_rect, image)  # 目标区域的左上角坐标（相对于search region的）
        # print(loc_pos)

        # delta_x and delta_y we want to estimate
        # delta = (np.array(loc_pos[:2]) - self._tmpl_sz / 2)
        delta = (np.array(loc_pos[:2]) - np.array(self._roi[2:]) / 2)  # 与原roi左上角坐标的相对差值
        # print('delta', delta)

        # scale between the search_roi and our template
        scale = loc_pos[2] * np.array(search_rect[2:]).astype(float) / (np.array(self._tmpl_sz) * 2)
        # back to the original size
        delta = delta * scale

        # print("delta * scale", delta)

        # add the delta to original position
        self._roi[0] = self._roi[0] + delta[0]
        self._roi[1] = self._roi[1] + delta[1]
        self._roi[2] = self._roi[2] * loc_pos[2]
        self._roi[3] = self._roi[3] * loc_pos[2]

        # some check boundary here
        if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        # update the template
        x = self.getFeatures(image, self._roi, self._tmpl_sz)  # new observation
        self.update_model(x, self._interp_factor)

        return self._roi

    def getSearchIdx(self, rang):
        ret = []
        step = int(rang/5)
        for i in range(0, step, 5):
            ret.append(i)
        for i in range(step, 2 * step, 3):
            ret.append(i)
        for i in range(2 * step, 3 * step):
            ret.append(i)
        for i in range(3 * step, 4 * step, 4):
            ret.append(i)
        for i in range(4 * step, 5 * step, 5):
            ret.append(i)
        return ret