import albumentations as A
import numpy as np

def salt_and_pepper() -> A.Compose:
    def _salt_and_pepper(image, amount=0.02):
        noisy = image.copy()
        total = image.shape[0] * image.shape[1]
        num_salt = int(amount * total / 2)
        num_pepper = int(amount * total / 2)

        # Salt
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255

        # Pepper
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0

        return noisy

    def apply_salt_and_pepper(img, amount=0.03, **kwargs): 
        return _salt_and_pepper(img, amount=amount)

    transform = A.Compose([
        A.Lambda(image=apply_salt_and_pepper),
    ])

    return transform