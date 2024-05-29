import cv2
import numpy as np
from PIL import Image
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from torch import nn

class FundusSegmentCrop(nn.Module):
    def __init__(self, predictor = None):
        if predictor:
            self.predictor = predictor
        else:
            model_type = "vit_t"
            sam_checkpoint = "./mobile_sam.pt"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            mobile_sam.to(device=device)
            mobile_sam.eval()
            self.predictor = SamPredictor(mobile_sam)

    def __call__(self, img):
        img_arr = np.array(img)

        # Annotate the image
        input_label, center_points = self.annotate(img_arr)

        # Generate the mask
        best_mask = self.generate_mask(img_arr, center_points, input_label)
        #convert the mask to np.unit8
        best_mask = best_mask.astype(np.uint8) * 255
        # Apply black thresholding
        best_mask = self.black_thresholding(img_arr, best_mask)

        # Extract contour from the mask
        contour = self.contour_from_mask(best_mask)

        # Create mask from the contour
        mask = self.mask_from_contour(img_arr, contour)

        # Crop the image and apply the mask
        img_cropped = self.crop_and_apply_mask(img_arr, mask)

        return Image.fromarray(img_cropped)
    
    def forward(self, img):
        return self.__call__(img)

    def annotate(self, img_arr, num_points=1000, num_corners=500, corner_offset=0.12):
        x_center = img_arr.shape[1] // 2
        y_center = img_arr.shape[0] // 2

        input_label = []
        center_points = []
        fundus_label = 1
        max_radius_x = int(0.7 * img_arr.shape[0] / 2)
        max_radius_y = int(0.7 * img_arr.shape[1] / 2)

        for j in range(num_points):
            radius_x = np.random.random() * max_radius_x
            radius_y = np.random.random() * max_radius_y
            angle = np.random.rand() * 2 * np.pi
            x = int(x_center + radius_x * np.cos(angle))
            y = int(y_center + radius_y * np.sin(angle))
            input_label.append(fundus_label)
            center_points.append([x, y])

        background_label = 0

        for j in range(num_corners):
            x_close_r_bottom = np.random.randint(int(img_arr.shape[1] * (1 - corner_offset)), img_arr.shape[1])
            y_close_r_bottom = np.random.randint(int(img_arr.shape[0] * (1 - corner_offset)), img_arr.shape[0])
            x_close_r_top = np.random.randint(int(img_arr.shape[1] * (1 - corner_offset)), img_arr.shape[1])
            y_close_r_top = np.random.randint(0, int(img_arr.shape[0] * corner_offset))
            x_close_l_bottom = np.random.randint(0, int(img_arr.shape[1] * corner_offset))
            y_close_l_bottom = np.random.randint(int(img_arr.shape[0] * (1 - corner_offset)), img_arr.shape[0])
            x_close_l_top = np.random.randint(0, int(img_arr.shape[1] * corner_offset))
            y_close_l_top = np.random.randint(0, int(img_arr.shape[0] * corner_offset))

            input_label.extend([background_label, background_label, background_label, background_label])
            center_points.extend([[x_close_l_top, y_close_l_top], [x_close_r_top, y_close_r_top], [x_close_l_bottom, y_close_l_bottom], [x_close_r_bottom, y_close_r_bottom]])

        return np.array(input_label), np.array(center_points)

    def generate_mask(self, img_arr, center_points, input_label):
        self.predictor.set_image(img_arr)
        masks, quality, _ = self.predictor.predict(point_coords=center_points, point_labels=input_label, multimask_output=True, return_logits=False)
        best_mask = masks[np.argmax(quality)]
        return best_mask

    def black_thresholding(self, img_arr, mask):
        black_pixels = np.all(img_arr < [4, 4, 4], axis=-1)
        mask[black_pixels] = 0
        return mask

    def contour_from_mask(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        return contour

    def mask_from_contour(self, img_arr, contour):
        mask = np.zeros_like(img_arr[:, :, 0]).astype(np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return mask

    def crop_and_apply_mask(self, img_arr, mask):
        colored_pixels = mask != 0
        x_max = np.max(np.where(colored_pixels)[0])
        x_min = np.min(np.where(colored_pixels)[0])
        y_max = np.max(np.where(colored_pixels)[1])
        y_min = np.min(np.where(colored_pixels)[1])
        img_cropped = img_arr[x_min:x_max, y_min:y_max]
        mask_cropped = mask[x_min:x_max, y_min:y_max]
        mask_cropped = np.expand_dims(mask_cropped, axis=-1)
        img_cropped = np.where(mask_cropped == 0, [0, 0, 0], img_cropped)
        return img_cropped.astype(np.uint8)
