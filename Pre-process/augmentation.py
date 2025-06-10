from datasets import load_from_disk
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import matplotlib.pyplot as plt
import random
import io

# Transform pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

def add_salt_and_pepper_noise(tensor, prob=0.03):
    noisy = tensor.clone()
    c, h, w = noisy.shape
    num_pixels = int(prob * h * w)
    for _ in range(num_pixels):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        value = random.choice([0.0, 1.0])
        noisy[:, y, x] = value
    return noisy

# Batch-wise augmentation
def batch_augment(batch):
    images = []
    for image in batch["image"]:
        tensor = transform(image)
        tensor = add_salt_and_pepper_noise(tensor, prob=0.01)
        pil_image = TF.to_pil_image(tensor)

        # Save PIL to bytes for HF compatibility
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        images.append({"bytes": buffer.read()})  # return bytes, not raw PIL

    return {"image": images}

# Load dataset
dataset = load_from_disk("/fred/oz402/aho/VLLM-MIA/Data/img_Flickr")
train_ds = dataset["train"] if "train" in dataset else dataset

# Apply batch-wise transformation
augmented_train = train_ds.map(
    batch_augment,
    batched=True,
    batch_size=32,             # Reduce this if still OOM, e.g., 8 or 16
    num_proc=1,                # You can increase this if memory allows
    desc="Augmenting dataset",
)

# Save the result
augmented_train.save_to_disk("/fred/oz402/aho/VLLM-MIA/Data/augmented_Flickr")

#---------------------------------------------------

# Get the original and augmented sample
original_image = train_ds[0]['image']
augmented_image = augmented_train[0]['image']

# Both are PIL already
original_image_pil = original_image
augmented_image_pil = augmented_image

# Show side-by-side comparison
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image_pil)
plt.axis('off')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image_pil)
plt.axis('off')
plt.title("Augmented")

plt.tight_layout()
plt.savefig("/fred/oz402/aho/VLLM-MIA/Pre-process/aug_comparison.png")
