import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class FeatureExtractor(nn.Module):
    def __init__(self, out_features=784):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # [8,256,16,16]
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # [8,128,8,8]
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),   # [8,64,4,4]
            nn.ReLU()
        )
        self.fc = nn.Linear(64*4*4, out_features)
    
    def forward(self, x):
        x = self.conv(x)                # [8, 64, 4, 4]
        x = x.flatten(1)                # [8, 1024]
        x = self.fc(x)                  # [8, 784]
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)  # output shape: [B, 32, 1, 1]
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x: [B, 512, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))     # [B, 128, 32, 32]
        x = F.relu(self.bn2(self.conv2(x)))     # [B, 32, 32, 32]
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, 32]
        x = F.relu(self.fc1(x))                # [B, 16]
        x = self.fc2(x)                        # [B, 1]
        return x

class PromptGenerator(nn.Module):
    def __init__(self, embed_dim=512, num_class=2):
        super().__init__()
        self.num_class = num_class
        self.prompt_embeddings = nn.Parameter(torch.zeros(num_class, embed_dim, 32, 32))

        self.inputs_head = FeatureExtractor()
        self.prompt_head = FeatureExtractor()
        self.head = Classifier()       
        self.criterion = nn.BCEWithLogitsLoss()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, class_ids=None):
        b, c, h, w = x.shape
        image_tokens = x
        prompt_tokens = self.prompt_embeddings
        # Embed the image and prompt tokens
        inputs_embed = self.inputs_head(image_tokens)
        prompt_embed = self.prompt_head(prompt_tokens)
        # prompt_embed = self.prompt_head(F.normalize(prompt_tokens, dim=-1))

        # Compute attention between image tokens and prompt tokens
        attn = torch.matmul(inputs_embed, prompt_embed.T)

        if class_ids is not None:
            # Direction loss based on class_ids
            mask_dots = F.one_hot(class_ids, num_classes=self.num_class).float()
            direction_loss = self.criterion(attn, mask_dots)

            # Selected prompts based on ground-truth class_ids
            selected_prompts = torch.einsum('bn,nchw->bchw', mask_dots, prompt_tokens) # [batch_size, C, H, W]

            # Add selected prompts to the image tokens
            x = torch.add(image_tokens, selected_prompts)

            # Classification loss
            logits = self.head(x).squeeze(-1)    
            cls_loss = self.criterion(logits, class_ids.float())

            prompt_loss = cls_loss + direction_loss
        else:
            prompt_loss = None
            # Select prompts based on the correlation Matrix with one-hot-max
            attn = F.gumbel_softmax(attn, tau=1.0, hard=True)  # [B, num_class]

            # Get the selected prompts
            selected_prompts = torch.einsum('bn,nchw->bchw', attn, prompt_tokens) # [batch_size, C, H, W]
            
            # Add selected prompts to the image tokens
            x = torch.add(image_tokens, selected_prompts)
        
        return x, prompt_loss
