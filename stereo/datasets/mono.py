# @Time    : 2024/10/27 02:00
# @Author  : zhangchenming
import numpy as np
import random
import cv2
from skimage.filters import gaussian, sobel
from scipy.interpolate import griddata
from torchvision import transforms


class WarpDataset:
    def __init__(self):
        self.max_disparity = 192
        self.feed_height = 352
        self.feed_width = 640
        self.process_width = self.feed_width + self.max_disparity
        self.xs, self.ys = np.meshgrid(np.arange(self.process_width), np.arange(self.feed_height))

        self.keep_aspect_ratio = True
        self.disable_sharpening = False
        self.disable_background = False

    def process_disparity(self, disparity, max_disparity_range=(40, 196)):
        """ Depth predictions have arbitrary scale - need to convert to a pixel disparity"""

        disparity = disparity.copy()

        # make disparities positive
        min_disp = disparity.min()
        if min_disp < 0:
            disparity += np.abs(min_disp)

        if random.random() < 0.01:
            # make max warped disparity bigger than network max -> will be clipped to max disparity,
            # but will mean network is robust to disparities which are too big
            max_disparity_range = (self.max_disparity * 1.05, self.max_disparity * 1.15)

        if disparity.max() == 0.0:
            disparity /= 1e-8  # now 0-1
        else:
            disparity /= disparity.max()  # now 0-1

        scaling_factor = (max_disparity_range[0] + random.random() *
                          (max_disparity_range[1] - max_disparity_range[0]))
        disparity *= scaling_factor

        if not self.disable_sharpening:
            # now find disparity gradients and set to nearest - stop flying pixels
            edges = sobel(disparity) > 3
            disparity[edges] = 0
            mask = disparity > 0

            try:
                disparity = griddata(np.stack([self.ys[mask].ravel(), self.xs[mask].ravel()], 1),
                                     disparity[mask].ravel(), np.stack([self.ys.ravel(),
                                                                        self.xs.ravel()], 1),
                                     method='nearest').reshape(self.feed_height, self.process_width)
            except (ValueError, IndexError) as e:
                pass  # just return disparity

        return disparity

    def project_image(self, image, disp_map, background_image):

        image = np.array(image)
        background_image = np.array(background_image)

        # set up for projection
        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        pix_locations = self.xs - disp_map

        # find where occlusions are, and remove from disparity map
        mask = self.get_occlusion_mask(pix_locations)
        masked_pix_locations = pix_locations * mask - self.process_width * (1 - mask)

        # do projection - linear interpolate up to 1 pixel away
        weights = np.ones((2, self.feed_height, self.process_width)) * 10000

        for col in range(self.process_width - 1, -1, -1):
            loc = masked_pix_locations[:, col]
            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                weight_up[mask]
            warped_image[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(self.feed_height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.

        # now fill occluded regions with random background
        if not self.disable_background:
            warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]

        warped_image = warped_image.astype(np.uint8)

        return warped_image

    def get_occlusion_mask(self, shifted):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(self.process_width - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
            _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

            mask_up[:, col] = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask

    def prepare_sizes(self, inputs):

        height, width, _ = np.array(inputs['left_image']).shape

        if self.keep_aspect_ratio:
            if self.feed_height <= height and self.process_width <= width:
                # can simply crop the image
                target_height = height
                target_width = width

            else:
                # check the constraint
                current_ratio = height / width
                target_ratio = self.feed_height / self.process_width

                if current_ratio < target_ratio:
                    # height is the constraint
                    target_height = self.feed_height
                    target_width = int(self.feed_height / height * width)

                elif current_ratio > target_ratio:
                    # width is the constraint
                    target_height = int(self.process_width / width * height)
                    target_width = self.process_width

                else:
                    # ratio is the same - just resize
                    target_height = self.feed_height
                    target_width = self.process_width

        else:
            target_height = self.feed_height
            target_width = self.process_width

        inputs = self.resize_all(inputs, target_height, target_width)

        # now do cropping
        if target_height == self.feed_height and target_width == self.process_width:
            # we are already at the correct size - no cropping
            pass
        else:
            self.crop_all(inputs)

        return inputs

    def crop_all(self, inputs):

        # get crop parameters
        height, width, _ = np.array(inputs['left_image']).shape
        top = int(random.random() * (height - self.feed_height))
        left = int(random.random() * (width - self.process_width))
        right, bottom = left + self.process_width, top + self.feed_height

        for key in ['left_image', 'background']:
            inputs[key] = inputs[key].crop((left, top, right, bottom))
        inputs['loaded_disparity'] = inputs['loaded_disparity'][top:bottom, left:right]

        return inputs

    @staticmethod
    def resize_all(inputs, height, width):

        # images
        img_resizer = transforms.Resize(size=(height, width))
        for key in ['left_image', 'background']:
            inputs[key] = img_resizer(inputs[key])
        # disparity - needs rescaling
        disp = inputs['loaded_disparity']
        disp *= width / disp.shape[1]

        disp = cv2.resize(disp.astype(float), (width, height))  # ensure disp is float32 for cv2
        inputs['loaded_disparity'] = disp

        return inputs
