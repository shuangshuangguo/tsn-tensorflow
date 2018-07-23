import copy
import os
import argparse
import pickle
import random
import json
import multiprocessing
import numpy as np
import cv2
import math
import time

from IPython import embed
from config import cfg


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class RTSPreprocessor(object):
    '''
        rgb frame distort with random crop and horizontal flip
        rgb_diff is calculated in this class
    '''
    fixed_h = 256
    fixed_w = 340
    patch_size = 224

    crop_density = 4 # position to crop
    max_distort = 1

    scales = [256, 224, 192, 168]
    crop_candi = []
    for i, _si in enumerate(scales):
        for j, _sj in enumerate(scales):
            if abs(i-j) <= max_distort:
                crop_candi.append((_si, _sj))


    @classmethod
    def get_distort_config(cls, validation=False):
        ''' generate a distort config for a segment '''
        if validation:
            h_start = (cls.fixed_h-cls.patch_size)//2
            w_start = (cls.fixed_w-cls.patch_size)//2
            h_end = min(h_start+cls.patch_size, cls.fixed_h)
            w_end = min(w_start+cls.patch_size, cls.fixed_w)
            return {'h_start':h_start, 'w_start':w_start,
                    'h_end':h_end, 'w_end':w_end, 'flip':False}

        (crop_w, crop_h) = cls.crop_candi[np.random.choice(range(len(cls.crop_candi)))]

        height_off = (cls.fixed_h-crop_h)//4
        width_off = (cls.fixed_w-crop_w)//4

        crop_pos = [(0, 0), (0, 4*width_off), (4*height_off, 0), (4*height_off, 4*width_off),
                    (2*height_off, 2*width_off), (0, 2 * width_off), (4*height_off, 2*width_off),
                    (2*height_off, 0), (2*height_off, 4*width_off), (1*height_off, 1*width_off),
                    (1*height_off, 3*width_off), (3*height_off, 1*width_off), (3*height_off, 3*width_off)]

        (h_start, w_start) = crop_pos[np.random.choice(range(len(crop_pos)))]

        h_end = min(h_start + crop_h, cls.fixed_h)
        w_end = min(w_start + crop_w, cls.fixed_w)

        # horizontal flip
        flip = np.random.random() < 0.5 and not validation

        return {'h_start':h_start, 'w_start':w_start, 'h_end':h_end, 'w_end':w_end, 'flip':flip}


    @classmethod
    def distort(cls, img, distort_config=None):
        ''' distort the images '''
        if distort_config is None:
            logger.warning('generate distort config by samples!')
            distort_config = cls.get_distort_config()

        # get distort configuration
        h_start = distort_config['h_start']
        w_start = distort_config['w_start']
        h_end = distort_config['h_end']
        w_end = distort_config['w_end']

        # resize 256 * 340
        fixed_resize = cv2.resize(img, (cls.fixed_w, cls.fixed_h), interpolation=cv2.INTER_LINEAR)
        # crop
        crop_im = fixed_resize[h_start:h_end, w_start:w_end, ]
        # resize 224 * 224
        dst = cv2.resize(crop_im, (cls.patch_size, cls.patch_size), interpolation=cv2.INTER_LINEAR)

        if len(dst.shape) == 2:
            dst = dst[:, :, np.newaxis]

        # flip the image
        if distort_config['flip']:
            dst = dst[:, ::-1, :]
        return dst

    @classmethod
    def process(cls, frames, rgb_distort_config):
        '''
            frames consists rgb_frame and its consecutive frames
            distort_config should be generated before calling the function
            images in a sequence should be implemented under one same config
        '''
        dis_frames = []

        for frame in frames:
            dis_frames.append(cls.distort(frame, rgb_distort_config))
        dis_frames = [d[np.newaxis,] for d in dis_frames]
        return np.concatenate(dis_frames, axis=0)



class TSNDataSet(data.Dataset):
    def __init__(self, list_file, num_segments=3, new_length=1,
                 modality='RGB', image_tmpl='img_{:05d}.jpg',
                 random_shift=True, test_mode=False):

        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.query = list(range(len(self.video_list)))
        np.random.shuffle(self.query)
        self._parse_list()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def batch_data(self):
        inputs, labels = [], []
        if len(self.query) < cfg.TRAIN.MINIBATCH:
            self.query = list(range(len(self.video_list)))
            np.random.shuffle(self.query)
        pick = self.query[:cfg.TRAIN.MINIBATCH]
        del self.query[:cfg.TRAIN.MINIBATCH]
        for idx in range(cfg.TRAIN.MINIBATCH):
            record = self.video_list[pick[idx]]
            if not self.test_mode:
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            else:
                segment_indices = self._get_test_indices(record)
            data, label = self.get(record, segment_indices)
            inputs.append(data)
            labels.append(label)
        inputs = np.concatenate(inputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        return inputs, labels


    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = cv2.imread(os.path.join(record.path, self.image_tmpl.format(p)))[:, :, ::-1]
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        distort_config = RTSPreprocessor.get_distort_config(self.test_mode)
        process_data = RTSPreprocessor.process(images, distort_config)
        process_data = process_data[np.newaxis,]
        return process_data, record.label


if __name__ == "__main__":
    dataset = TSNDataSet(list_file, num_segments=3, new_length=1,
                modality='RGB', image_tmpl='img_{:05d}.jpg',
                random_shift=True, test_mode=False)


