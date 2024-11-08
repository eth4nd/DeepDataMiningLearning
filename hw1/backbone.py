import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock

class CustomLastLevelMaxPool(ExtraFPNBlock):
    """
    Custom pooling class that mimics LastLevelMaxPool behavior, inheriting from ExtraFPNBlock
    to be compatible with FeaturePyramidNetwork.
    """
    def forward(self, x, y, names):
        # Print type of x for debugging purposes
        print("Type of x:", type(x))
        
        # Handle different input types for x
        if isinstance(x, dict):
            last_feature = list(x.values())[-1]  # Get the last feature map
        elif isinstance(x, list):
            last_feature = x[-1]  # Use the last element of the list as the feature map
        elif isinstance(x, torch.Tensor):
            last_feature = x  # Directly use the tensor if it's already in tensor form
        else:
            raise TypeError("Expected x to be a dictionary, list, or tensor, but got a different type:", type(x))

        # Apply max pooling to the last feature map
        pooled = F.max_pool2d(last_feature, kernel_size=2, stride=2)

        if isinstance(x, dict):
            x["pool"] = pooled  # Add the pooled feature to the dictionary
            names.append("pool")
        elif isinstance(x, list):
            x.append(pooled)  # Add the pooled feature to the list
            names.append("pool")
        else:
            # If x is a tensor, return the pooled output as a new feature
            x = {"pool": pooled}
            names = ["pool"]

        return x, names

def get_efficientnet_backbone(pretrained=True):
    """
    Initializes the EfficientNet-B0 backbone model with or without pretrained weights
    and removes the classification head for feature extraction.
    """
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)
    else:
        backbone = efficientnet_b0(weights=None)
    
    # Remove the classification head to retain only the feature extraction layers
    backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove last two layers to keep features
    return backbone

class CustomBackboneWithFPN(nn.Module):
    """
    Custom backbone model integrated with a Feature Pyramid Network (FPN) for multi-scale feature extraction
    using custom forward hooks to extract features from nested layers.
    """
    def __init__(self, backbone: nn.Module, layer_names: list, out_channels: int = 256):
        super(CustomBackboneWithFPN, self).__init__()
        self.backbone = backbone
        self.feature_outputs = {}
        self.layer_names = layer_names

        # Register hooks for the specified layers
        for name, layer in self.backbone.named_modules():
            if name in self.layer_names:
                layer.register_forward_hook(self._hook_fn(name))

        # Update the FPN to use the correct in_channels_list based on actual outputs
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[16, 24, 80, 192],  # Use the actual output channel sizes
            out_channels=out_channels,
            extra_blocks=CustomLastLevelMaxPool()  # Use the custom pooling class
        )
    
    def _hook_fn(self, name):
        def hook(module, input, output):
            # Save the output of the layer to the feature outputs dictionary
            self.feature_outputs[name] = output
        return hook

    def forward(self, x, targets=None):
        """
        Forward method modified to accept targets for compatibility with object detection models.
        """
        # Ensure x is a Tensor before passing it through the backbone
        if isinstance(x, list):
            # Convert the list of Tensors to a batch Tensor
            x = torch.stack(x).to(next(self.backbone.parameters()).device)

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a Tensor, but got {type(x)}")

        # Clear previous feature outputs
        self.feature_outputs.clear()

        # Pass the input through the backbone to trigger the hooks
        _ = self.backbone(x)

        # Collect the feature outputs in the correct order
        features = [self.feature_outputs[name] for name in self.layer_names]

        # Debugging: Print shapes of extracted features
        for i, feature in enumerate(features):
            print(f"[DEBUG] Feature {i} ({self.layer_names[i]}): shape {feature.shape}")

        # Create a dictionary to pass to the FPN
        feature_dict = {str(i): feature for i, feature in enumerate(features)}

        # Pass the features through the FPN
        fpn_output = self.fpn(feature_dict)

        # Ensure the output is a Tensor
        if not all(isinstance(feature, torch.Tensor) for feature in features):
            raise TypeError("One or more feature outputs are not Tensors")

        # Return fpn_output and targets if provided (for compatibility with detection models)
        if self.training and targets is not None:
            return fpn_output, targets

        return fpn_output

    
if __name__ == "__main__":
    # Initialize the EfficientNet-B0 backbone with pretrained weights
    backbone = get_efficientnet_backbone(pretrained=True)

    # Specify the layer names to hook based on your model structure
    layer_names = [
        "0.1.0.block",
        "0.2.1.block",
        "0.4.2.block",
        "0.6.3.block"
    ]

    # Create an instance of the CustomBackboneWithFPN using EfficientNet-B0
    model = CustomBackboneWithFPN(backbone, layer_names=layer_names, out_channels=256)

    # Test with a sample tensor to ensure correct output shapes
    x = torch.rand(1, 3, 256, 256)  # Adjust the input size as needed
    output = model(x)
    print([(k, v.shape) for k, v in output.items()])
