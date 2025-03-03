# %%
import os
from PIL import Image
import random

# Paths to your image and mask folders
image_folder = 'C:/Users/krish/Downloads/dataset RITE/dataset RITE/images'
mask_folder = 'C:/Users/krish/Downloads/dataset RITE/dataset RITE/av'
augmented_image_folder = 'C:/Users/krish/Downloads/dataset RITE/dataset RITE/augmented_images'
augmented_mask_folder = 'C:/Users/krish/Downloads/dataset RITE/dataset RITE/augmented_masks'

# Create folders to save augmented images and masks if they do not exist
os.makedirs(augmented_image_folder, exist_ok=True)
os.makedirs(augmented_mask_folder, exist_ok=True)

# Data augmentation function
def augment_and_save(image_path, mask_path, count=10):
    original_image = Image.open(image_path)
    original_mask = Image.open(mask_path)

    for i in range(count):
        image = original_image.copy()
        mask = original_mask.copy()

        # Randomly choose augmentation techniques
        if random.random() < 0.5:
            # Flip horizontally
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            # Flip vertically
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            # Rotate 90 degrees
            image = image.rotate(90)
            mask = mask.rotate(90)

        # Save augmented images and masks in their respective formats
        image_name = os.path.basename(image_path).split('.')[0]
        mask_name = os.path.basename(mask_path).split('.')[0]

        # Save the augmented image as .tif and the mask as .png
        augmented_image_path = os.path.join(augmented_image_folder, f"{image_name}_aug_{i}.tif")
        augmented_mask_path = os.path.join(augmented_mask_folder, f"{mask_name}_aug_{i}.png")

        image.save(augmented_image_path)
        mask.save(augmented_mask_path)

        print(f"Saved augmented image: {augmented_image_path}")
        print(f"Saved augmented mask: {augmented_mask_path}")

# Initial counts of original images and masks
original_images = os.listdir(image_folder)
original_masks = os.listdir(mask_folder)

print(f"Original number of images: {len(original_images)}")
print(f"Original number of masks: {len(original_masks)}")

# Loop through all images and masks
for image_name in original_images:
    # Corresponding mask name (assumes mask has the same name as image but different extension)
    mask_name = image_name.replace('.tif', '.png')  # Adjust if necessary based on your filenames
    image_path = os.path.join(image_folder, image_name)
    mask_path = os.path.join(mask_folder, mask_name)

    if mask_name in original_masks:  # Check if the corresponding mask exists
        augment_and_save(image_path, mask_path, count=10)
        print(f"Augmented {image_name} and {mask_name}")

# Count of augmented images and masks
augmented_images = os.listdir(augmented_image_folder)
augmented_masks = os.listdir(augmented_mask_folder)

print(f"Augmented number of images: {len(augmented_images)}")
print(f"Augmented number of masks: {len(augmented_masks)}")

print("Data augmentation complete.")


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Downward path (Encoder)
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature

        # Upward path (Decoder)
        for feature in reversed(features):
            self.decoder.append(self._upconv(feature * 2, feature))
            self.decoder.append(self._conv_block(feature * 2, feature))

        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the skip connections

        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Up-convolution
            skip_connection = skip_connections[idx // 2]

            # If necessary, pad x to match the skip_connection size
            if x.shape != skip_connection.shape:
                x = F.pad(x, [0, skip_connection.shape[3] - x.shape[3], 0, skip_connection.shape[2] - x.shape[2]])

            x = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection
            x = self.decoder[idx + 1](x)  # Convolution after concatenation

        return self.final_conv(x)

# Example usage:
unet = UNet(in_channels=3, out_channels=1).to('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn((1, 3, 256, 256)).to('cuda' if torch.cuda.is_available() else 'cpu')
output = unet(x)
print(f"Output shape: {output.shape}")


# %%
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []
        for feature in features:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=2, padding=1))  # Final output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Example usage:
discriminator = Discriminator(in_channels=1).to('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn((1, 1, 256, 256)).to('cuda' if torch.cuda.is_available() else 'cpu')
output = discriminator(x)
print(f"Discriminator output shape: {output.shape}")


# %%
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


class FundusDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)  # Assuming same number of images and masks

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB for images
        mask = Image.open(mask_path).convert('L')      # Ensure 1-channel for masks

        # Optionally apply any transformations here
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert images to PyTorch tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return image, mask  # Return both original and mask images

# Specify the folder paths for images and masks
image_folder = 'C:/Users/krish/Downloads/dataset RITE/dataset RITE/final_images'  # Update this to your actual path
mask_folder = 'C:/Users/krish/Downloads/dataset RITE/dataset RITE/final_masks'  # Update this to your actual path

# Create the dataset and dataloader
dataset = FundusDataset(image_folder, mask_folder)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Continue with your GAN training code...


# Initialize the models
generator = UNet(in_channels=3, out_channels=1).to('cuda' if torch.cuda.is_available() else 'cpu')
discriminator = Discriminator(in_channels=1).to('cuda' if torch.cuda.is_available() else 'cpu')

# Define optimizers for both the generator and discriminator
gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# Define loss functions
adversarial_loss = nn.BCEWithLogitsLoss()  # Binary cross-entropy for real vs fake classification
segmentation_loss = nn.BCEWithLogitsLoss()  # Segmentation loss

# Define training loop
def train_gan(dataloader, epochs=100):
    for epoch in range(epochs):
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            masks = masks.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Train Discriminator
            disc_optimizer.zero_grad()

            # Real masks
            real_preds = discriminator(masks)
            real_targets = torch.ones_like(real_preds)  # Target labels are 1 for real masks
            real_loss = adversarial_loss(real_preds, real_targets)

            # Fake masks
            fake_masks = generator(images)
            fake_preds = discriminator(fake_masks.detach())  # Detach to avoid backprop through generator
            fake_targets = torch.zeros_like(fake_preds)  # Target labels are 0 for fake masks
            fake_loss = adversarial_loss(fake_preds, fake_targets)

            # Discriminator loss and update
            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            disc_optimizer.step()

            # Train Generator
            gen_optimizer.zero_grad()

            # Adversarial loss for the generator
            fake_preds = discriminator(fake_masks)
            gen_adv_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds))  # Wants to fool the discriminator

            # Segmentation loss for the generator
            gen_seg_loss = segmentation_loss(fake_masks, masks)

            # Total generator loss and update
            gen_loss = gen_adv_loss + gen_seg_loss
            gen_loss.backward()
            gen_optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Gen Loss: {gen_loss.item():.4f} | Disc Loss: {disc_loss.item():.4f}")

# Example usage: Start training
train_gan(dataloader, epochs=4)


# %%
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def visualize_masks(generator, dataloader, epoch, device='cuda'):
    generator.eval()  # Set the generator to evaluation mode
    images, masks = next(iter(dataloader))  # Get a batch from the dataloader
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        generated_masks = generator(images)

    # Convert the tensors to CPU for plotting
    images = images.cpu()
    masks = masks.cpu()
    generated_masks = generated_masks.cpu()

    # Create a grid to visualize the results
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
    fig.suptitle(f"Epoch {epoch} - Ground Truth vs Generated Masks", fontsize=16)

    for i in range(4):
        # Original image
        axes[0, i].imshow(transforms.ToPILImage()(images[i]))
        axes[0, i].set_title('Image')
        axes[0, i].axis('off')

        # Ground truth mask
        axes[1, i].imshow(masks[i][0], cmap='gray')
        axes[1, i].set_title('Ground Truth Mask')
        axes[1, i].axis('off')

        # Generated mask
        axes[2, i].imshow(generated_masks[i][0], cmap='gray')
        axes[2, i].set_title('Generated Mask')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    generator.train()  # Set the generator back to training mode

def train_gan_with_visualization(dataloader, epochs=100, device='cuda'):
    for epoch in range(epochs):
        generator.train()
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Train Discriminator
            disc_optimizer.zero_grad()
            real_preds = discriminator(masks)
            real_targets = torch.ones_like(real_preds)
            real_loss = adversarial_loss(real_preds, real_targets)

            fake_masks = generator(images)
            fake_preds = discriminator(fake_masks.detach())
            fake_targets = torch.zeros_like(fake_preds)
            fake_loss = adversarial_loss(fake_preds, fake_targets)

            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            disc_optimizer.step()

            # Train Generator
            gen_optimizer.zero_grad()
            fake_preds = discriminator(fake_masks)
            gen_adv_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds))
            gen_seg_loss = segmentation_loss(fake_masks, masks)
            gen_loss = gen_adv_loss + gen_seg_loss
            gen_loss.backward()
            gen_optimizer.step()

        # Print losses after each epoch
        print(f"Epoch [{epoch+1}/{epochs}] | Gen Loss: {gen_loss.item():.4f} | Disc Loss: {disc_loss.item():.4f}")

        # Visualize generated masks after each epoch
        visualize_masks(generator, dataloader, epoch + 1, device)

# Example usage: Start training with visualization
train_gan_with_visualization(dataloader, epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu')


# %%
import torch

def evaluate_segmentation(generator, dataloader, device='cuda', threshold=0.5):
 
    generator.eval()  
    dice_scores, iou_scores, accuracies = [], [], []
    precisions, recalls = [], []

    smoothing = 1e-6  

    with torch.no_grad():  
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            
    
            outputs = generator(images)
            outputs = torch.sigmoid(outputs)  
            predictions = (outputs > threshold).float() 

            # Compute metrics
            intersection = (predictions * masks).sum(dim=(2, 3)) 
            union = (predictions + masks).sum(dim=(2, 3)) - intersection

            dice = (2 * intersection + smoothing) / (predictions.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + smoothing)
            iou = (intersection + smoothing) / (union + smoothing)
            accuracy = ((predictions == masks).float().sum(dim=(2, 3)) + smoothing) / (predictions.shape[2] * predictions.shape[3])
            precision = (intersection + smoothing) / (predictions.sum(dim=(2, 3)) + smoothing)
            recall = (intersection + smoothing) / (masks.sum(dim=(2, 3)) + smoothing)

            # Append metrics for each batch
            dice_scores.append(dice.mean().item())
            iou_scores.append(iou.mean().item())
            accuracies.append(accuracy.mean().item())
            precisions.append(precision.mean().item())
            recalls.append(recall.mean().item())

    # Compute average metrics
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_accuracy = (sum(accuracies) / len(accuracies))
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    # Print the results
    print(f"Dice Score: {avg_dice:.4f}")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")

    # Return metrics as a dictionary
    return {
        "Dice Score": avg_dice,
        "IoU": avg_iou,
        "Accuracy": avg_accuracy,
        "Precision": avg_precision,
        "Recall": avg_recall
    }
# Assuming `generator` is your trained model, and `dataloader` is the DataLoader for evaluation data.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator.to(device)

# Evaluate the model
results = evaluate_segmentation(generator, dataloader, device=device, threshold=0.5)

# Print the results
print("Evaluation Results:", results)



