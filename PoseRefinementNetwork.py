import torch
import torch.nn as nn
import torchvision.models as models

class PoseRefinementNetwork(nn.Module):
    def __init__(self):
        super(PoseRefinementNetwork, self).__init__()
        # Load a pretrained ResNet model
        self.feature_extractor = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Remove the final fully connected layer
        self.feature_extractor.fc = nn.Identity()
        
        # Transformer encoder for processing the features
        # Adjust d_model to match the concatenated feature size (512 * 2 = 1024)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output layers for pose updates
        # Adjust input features to 1024 to match the output of the transformer encoder
        self.fc_translation = nn.Linear(1024, 3)  # Adjusted for translation update
        self.fc_rotation = nn.Linear(1024, 4)  # Adjusted for rotation update (quaternion representation)

    def forward(self, rendered_img, real_img_cropped):
        # Extract features from both images
        rendered_features = self.feature_extractor(rendered_img)
        real_features = self.feature_extractor(real_img_cropped)
        
        # Concatenate features and prepare for transformer
        combined_features = torch.cat((rendered_features, real_features), dim=1)
        combined_features = combined_features.unsqueeze(0)  # Add batch dimension for transformer
        
        # Pass through transformer encoder
        transformed_features = self.transformer_encoder(combined_features)
        
        # Predict translation and rotation updates
        translation_update = self.fc_translation(transformed_features.squeeze(0))
        rotation_update = self.fc_rotation(transformed_features.squeeze(0))
        
        return translation_update, rotation_update
    

import torch
import torch.nn as nn
import torchvision.models as models

class PoseRefinementNetworkWithTransformer(nn.Module):
    def __init__(self):
        super(PoseRefinementNetworkWithTransformer, self).__init__()
        # Load a pretrained ResNet model
        self.feature_extractor = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer of the ResNet model to accept 1-channel input instead of the default 3
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Remove the final fully connected layer to use ResNet as a feature extractor
        self.feature_extractor.fc = nn.Identity()
        
        # Reduce the combined feature size from 1024 (512*2) to 512 to match the transformer's expected input size
        self.feature_size_reducer = nn.Linear(1024, 512)
        
        # Define Transformer Encoder Layer with d_model=512
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output layers for pose updates
        self.fc_translation = nn.Linear(512, 3)
        self.fc_rotation = nn.Linear(512, 4)

    def forward(self, rendered_img, real_img_cropped):
        # Extract features from both images using the modified ResNet as the feature extractor
        rendered_features = self.feature_extractor(rendered_img).flatten(start_dim=1)
        real_features = self.feature_extractor(real_img_cropped).flatten(start_dim=1)
        
        # Concatenate features from both images
        combined_features = torch.cat((rendered_features, real_features), dim=1)
        
        # Reduce the combined feature size to match the transformer's expected input size
        reduced_features = self.feature_size_reducer(combined_features)
        
        # Add an extra dimension for the transformer
        reduced_features = reduced_features.unsqueeze(1)
        
        # Transformer encoder with residual connection
        # Adding the original reduced features to its transformed version
        transformed_features = self.transformer_encoder(reduced_features)
        # Ensure the original reduced_features is broadcastable to the transformed_features shape
        residual_connection = reduced_features + transformed_features
        
        # Remove the sequence dimension
        final_features = residual_connection.squeeze(1)
        
        # Predict translation and rotation updates
        translation_update = self.fc_translation(final_features)
        rotation_update = self.fc_rotation(final_features)
        
        return translation_update, rotation_update

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

class PoseRefinementNetworkWithViT(nn.Module):
    def __init__(self):
        super(PoseRefinementNetworkWithViT, self).__init__()
        
        # Initialize a Vision Transformer model with pretrained weights
        self.feature_extractor = vit_b_16(pretrained=True)
        
        # Remove the classifier head to use the ViT as a feature extractor
        self.feature_extractor.head = nn.Identity()
        
        # Adjust the input dimensions of the linear layers to match the actual feature size
        feature_size = 1000  # Adjusted based on diagnostic output
        
        # Output layers for pose updates
        self.fc_translation = nn.Linear(feature_size, 3)
        self.fc_rotation = nn.Linear(feature_size, 4)

    def forward(self, rendered_img, real_img_cropped):
        # Process images and extract features as before
        rendered_img = self.prepare_image(rendered_img)
        real_img_cropped = self.prepare_image(real_img_cropped)
        
        rendered_features = self.feature_extractor(rendered_img)
        real_features = self.feature_extractor(real_img_cropped)
        
        # Now, correctly aggregate and predict updates based on adjusted feature size
        translation_update = (self.fc_translation(rendered_features) + self.fc_translation(real_features)) / 2
        rotation_update = (self.fc_rotation(rendered_features) + self.fc_rotation(real_features)) / 2
        
        return translation_update, rotation_update
    
    def prepare_image(self, img):
        # Ensure the image has 3 channels by repeating the single channel
        img_3ch = img.repeat(1, 3, 1, 1)
        # Resize the image to match the expected input size of the ViT model (224x224)
        img_resized = F.interpolate(img_3ch, size=(224, 224), mode='bilinear', align_corners=False)
        return img_resized

import torch
import torch.nn as nn
import torchvision.models as models

class PoseRefinementNetworkSimple(nn.Module):
    def __init__(self):
        super(PoseRefinementNetworkSimple, self).__init__()
        # Load a pretrained ResNet model
        self.feature_extractor = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Remove the final fully connected layer
        self.feature_extractor.fc = nn.Identity()
        
        # Instead of using a transformer, directly concatenate the features and use a linear layer
        # Assuming that the feature size is 512 for each image from ResNet18 and we concatenate them
        self.fc_combined = nn.Linear(512 * 2, 512)  # Combined feature layer
        
        # Output layers for pose updates, directly from the combined features
        self.fc_translation = nn.Linear(512, 3)  # Adjusted for translation update
        self.fc_rotation = nn.Linear(512, 4)  # Adjusted for rotation update (quaternion representation)

    def forward(self, rendered_img, real_img_cropped):
        # Extract features from both images
        rendered_features = self.feature_extractor(rendered_img).flatten(start_dim=1)
        real_features = self.feature_extractor(real_img_cropped).flatten(start_dim=1)
        
        # Concatenate features
        combined_features = torch.cat((rendered_features, real_features), dim=1)
        
        # Pass concatenated features through a combined feature layer
        combined_features = self.fc_combined(combined_features)
        
        # Predict translation and rotation updates
        translation_update = self.fc_translation(combined_features)
        rotation_update = self.fc_rotation(combined_features)
        
        return translation_update, rotation_update