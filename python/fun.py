#!/usr/bin/env python
# coding=utf8

# Copyright (c) 2018 Behrooz Vedadian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Some fun transformations applied to a sample image
"""

from __future__ import print_function, division, unicode_literals

import math
import codecs
import sys
import cv2
import numpy as np
from matplotlib import pyplot
import math

# Make the program look alike in python versions 3 and 2
if sys.version_info < (3, 0):
    sys.stdin = codecs.getreader("utf8")(sys.stdin)
    sys.stdout = codecs.getwriter("utf8")(sys.stdout)
    sys.stderr = codecs.getwriter("utf8")(sys.stderr)

def main():
    
    def show(*images, **kwargs):
        fig = pyplot.figure() 
        if 'title' in kwargs:
            fig.canvas.set_window_title(kwargs['title']) 
        if 'do_not_share_axes' in kwargs:
            do_not_share_axes = True
        else:
            do_not_share_axes = False
        r = math.floor(math.sqrt(len(images)))
        c = math.ceil(len(images) / r)
        i = 1
        shared_axes = None
        for image in images:
            if isinstance(image, tuple):
                image, title = image
            else:
                title = ''
            if r > 1 or c > 1:
                axes = pyplot.subplot(r, c, i, sharex=shared_axes, sharey=shared_axes)
                if shared_axes is None and not do_not_share_axes:
                    shared_axes = axes
            if title:
                pyplot.title(title)
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pyplot.imshow(image, cmap='gray')
            pyplot.xticks([])
            pyplot.yticks([])
            i += 1
        
        pyplot.show()

    def compute_energy_matrix(image):
        # sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        # energy_matrix = np.abs(sobel_x) + np.abs(sobel_y)
        energy_matrix = np.abs(cv2.Laplacian(image, cv2.CV_32F))
        energy_matrix = np.sum(energy_matrix, axis=2)
        return energy_matrix

    def find_vertical_seam(energy_matrix):
        max_deviation = 2
        m, n = energy_matrix.shape
        path_energy = np.zeros((m, n + 2 * max_deviation), np.float32) + sys.float_info.max
        path_energy[0, max_deviation:-max_deviation] = energy_matrix[0, :]
        offsets = np.zeros(energy_matrix.shape)
        for i in range(1, m):
            offset_costs = np.tile(energy_matrix[i, :].reshape(1, -1), (2 * max_deviation + 1, 1))
            for o in range(2 * max_deviation + 1):
                offset_costs[o, :] += path_energy[i - 1, o:o+n]
            offsets[i, :] = np.argmin(offset_costs, axis=0) - max_deviation
            path_energy[i, max_deviation:-max_deviation] = np.min(offset_costs, axis=0)
        seam = np.zeros(m, np.int)
        seam[m - 1] = np.argmin(path_energy[m - 1, max_deviation:-max_deviation])
        for i in range(m - 1, 0, -1):
            seam[i - 1] = seam[i] + offsets[i, seam[i]]
        return seam

    def remove_vertical_seam(image, seam):
        m, n = image.shape[:2]
        for i in range(m):
            if seam[i] < n - 1:
                image[i, seam[i]:-1] = image[i, seam[i]+1:]
        image = image[:, 0:n - 1]
        return image

    def update_seams_offset(seams_offset, seam):
        m, n = seams_offset.shape
        for i in range(m):
            seams_offset[i, seam[i]:] += 1
        return seams_offset

    def draw_seam(seams_overlay, seams_offset, seam):
        for i in range(seams_overlay.shape[0]):
            seams_overlay[i, seam[i] + seams_offset[i, seam[i]], :] = (0, 255, 0)

    image = cv2.imread('../penguines.jpg')

    h, w = image.shape[:2]

    seams_overlay = np.copy(image)
    seams_offset = np.zeros(image.shape[:2], np.int)

    for i in range(250):
        energy_matrix = compute_energy_matrix(image)
        vertical_seam = find_vertical_seam(energy_matrix)
        image = remove_vertical_seam(image, vertical_seam)
        draw_seam(seams_overlay, seams_offset, vertical_seam)
        print('{} seem(s) removed.'.format(i + 1))
    

    show(np.concatenate([seams_overlay, np.zeros((h, 20, 3), np.uint8), image], axis=1), title='Seam carving', do_not_share_axes=True)

if __name__ == "__main__":
    main()
