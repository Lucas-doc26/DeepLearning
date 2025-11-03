import tensorflow as tf
import numpy as np
import albumentations as A
import cv2

def albumentations_tf(img, transform:A.Compose, prob:float=0.7):
    def _apply(img_np):
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        out = transform(image=img_np)['image']
        return out.astype(np.float32) / 255.0

    do_aug = tf.less(tf.random.uniform([], 0, 1.0), prob) # probabilidade de aplicar a transformação
    img = tf.cond(do_aug,
                  lambda: tf.numpy_function(_apply, [img], tf.float32),
                  lambda: img)
    return img

def apply_rain(img):
    # garante uint8 pra Albumentations funcionar
    img_up = (img * 255).astype(np.uint8) if img.dtype!=np.uint8 else img
    img_up = cv2.resize(img_up, (256, 256), cv2.INTER_LINEAR)
    rain = A.RandomRain(
            slant_range=(-10, 10),       # ângulo aleatório entre -10° e +10°
            drop_length=10,
            drop_width=3,
            drop_color=(50, 50, 50),
            blur_value=5,
            brightness_coefficient=0.8,
            p=1
        )    
    img_rain = rain(image=img_up)['image']
    img_down = cv2.resize(img_rain, (64, 64), cv2.INTER_AREA)
    return img_down.astype(np.float32) / 255.0

def apply_contrast(img):
    contrast = A.AutoContrast(cutoff=0, method="cdf", p=1)
    return albumentations_tf(img, contrast)

def apply_defocus(img):
    defocus = A.Defocus(radius=[3,10], alias_blur=[0.1,0.5],p=1)
    return albumentations_tf(img, defocus)

def apply_brightness_contrast(img):
    rbc = A.RandomBrightnessContrast(brightness_limit=[-0.8,0], contrast_limit=[-0.3,0.2], p=1)
    return albumentations_tf(img, rbc)

def apply_rotate_upside_down(img):
    upside = A.Rotate(limit=[179,180], border_mode=cv2.BORDER_REPLICATE, fill=0, p=1)
    return albumentations_tf(img, upside)

def apply_gaussian_noise(img, mean=0.0, std=0.1):
    img_f = img.astype(np.float32)
    noise = np.random.normal(mean, std, img_f.shape)
    noisy = np.clip(img_f + noise, 0, 1)
    return noisy

def apply_channel_shuffle(img):
    channel_shuffle = A.ChannelShuffle(p=1)
    return albumentations_tf(img, channel_shuffle)

def randon_rain():
    return A.Compose([
            A.RandomRain(
                drop_length=8, drop_width=1,
                drop_color=(180, 180, 180),  blur_value=5,brightness_coefficient=0.8, p=0.15
            ),
            A.GaussNoise(var_limit=(0.0, 0.0007), mean=0, p=0.15),
            A.ChannelShuffle(p=0.15),
            A.Rotate(limit=40, p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,brightness_by_max=True, p=0.15),
            A.AdvancedBlur(blur_limit=(7,9), noise_limit=(0.75, 1.25), p=0.15),
            #A.Resize(height=256, width=256)
    ])