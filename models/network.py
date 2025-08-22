import torch
from torch import nn
from einops import repeat, rearrange
import torch.nn.functional as F
from torchvision.utils import save_image

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class Network(nn.Module):    
    def __init__(self, source_encoder, target_encoder, ST_module, CPG_module, decoder, vgg, device, mask=True):

        super().__init__()
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.ST_module = ST_module
        self.CPG_module = CPG_module
        self.decoder = decoder

        self.device = device

        self.mask = mask

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, batch_source, batch_target):
        source_imgs = batch_source["img"].to(self.device)

        B = source_imgs.size(0)

        target_imgs = batch_target["img"].to(self.device)        
        target_labels = batch_target["mask"].to(self.device) if self.mask else batch_target["cls_label"].to(self.device)

        target_imgs = target_imgs[:B]
        target_labels = target_labels[:B]

        ### 1. Feature Extractor ###
        pos_embedding = self.source_encoder.pos_embedding
        source_c_feats = self.source_encoder(source_imgs)
        target_s_feats = self.target_encoder(target_imgs)

        ### 2. Style Transfer Module ###
        hst_feats = self.ST_module(source_c_feats, target_s_feats, pos_embedding)

        ### 3. Class-aware Prompt Generator ###
        b, c, h, w = hst_feats.shape            
        hst_feats, _ = self.CPG_module(hst_feats)

        ### 4. Decoder ###
        Ist = self.decoder(hst_feats)

        ### Perceptual features ###
        source_vgg = self.encode_with_intermediate(source_imgs)
        target_vgg = self.encode_with_intermediate(target_imgs)
        Ist_vgg = self.encode_with_intermediate(Ist)

        # Content loss
        loss_c = self.calc_content_loss(Ist_vgg[-1], source_vgg[-1]) + self.calc_content_loss(Ist_vgg[-2], source_vgg[-2])
        # Style loss
        loss_s = self.calc_style_loss(Ist_vgg[0], target_vgg[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ist_vgg[i], target_vgg[i])

        # Contrastive loss
        ht = rearrange(target_s_feats, "(h w) b c -> b c h w", h=h, w=w)
        # target_labels = repeat(target_labels, "b h w-> b (h w)", h=h, w=w) if self.mask else repeat(target_labels, "b -> b (h w)", h=h, w=w)
        _, loss_prompt = self.CPG_module(ht, target_labels)

        return Ist, loss_c, loss_s, loss_prompt
