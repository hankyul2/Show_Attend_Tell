import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class Encoder(nn.Module):
    def __init__(self, model_name, pretrained=False, output_size=(14, 14)):
        super(Encoder, self).__init__()
        self.ConvNet = torch.hub.load('hankyul2/EfficientNetV2-pytorch', model_name, pretrained=pretrained)
        self.ConvNet.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size), Rearrange('b c h w -> b h w c'))

    def forward(self, x):
        return self.ConvNet(x)


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super(Attention, self).__init__()
        self.en = nn.Linear(encoder_dim, attn_dim)
        self.de = nn.Linear(decoder_dim, attn_dim)
        self.at = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, en, de):
        en_attn, de_attn = self.en(en), self.de(de)
        alpha = F.softmax(self.at(self.relu(en_attn + de_attn.unsqueeze(dim=1))).squeeze(-1), dim=1)
        alpha_weighted_en = (alpha.unsqueeze(-1) * en).sum(dim=1)
        return alpha_weighted_en, alpha


class Decoder(nn.Module):
    def __init__(self, encoder_dim, embed_dim, decoder_dim, attn_dim, vocab_size=100, dropout=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.attn = Attention(encoder_dim, decoder_dim, attn_dim)
        self.gate = nn.Sequential(nn.Linear(decoder_dim, encoder_dim), nn.Sigmoid())
        self.decoder = nn.LSTMCell(encoder_dim + embed_dim, decoder_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(decoder_dim, vocab_size))

    def forward(self, encoder_out, encoded_text, encoded_text_len):
        device = encoder_out.device
        img = rearrange(encoder_out, 'b h w c -> b (h w) c')
        batch_size, pixel_size, vocab_size = encoder_out.size(0), img.size(1), self.vocab_size

        h, c = self.init_h_c(img)
        text_len, idx = encoded_text_len.squeeze(1).sort(dim=0, descending=True)
        img, text, text_len = img[idx], self.embed(encoded_text[idx]), (text_len - 1).tolist()

        prediction = torch.zeros(batch_size, max(text_len), vocab_size).to(device)
        attention = torch.zeros(batch_size, max(text_len), pixel_size).to(device)

        for t in range(max(text_len)):
            batch_size_t = sum(s > t for s in text_len)
            img_t, text_t = img[:batch_size_t], text[:batch_size_t, t]
            h_t, c_t = h[:batch_size_t], c[:batch_size_t]

            attn_weighted_img, attn = self.attn(img_t, h_t)
            attn_weighted_img *= self.gate(h_t)
            h, c = self.decoder(torch.cat([attn_weighted_img, text_t], dim=1), (h_t, c_t))

            prediction[:batch_size_t, t, :] = self.head(h)
            attention[:batch_size_t, t, :] = attn

        return prediction, encoded_text[idx], text_len, attention, idx

    def init_h_c(self, img):
        img_mean = img.mean(dim=1)
        return self.init_h(img_mean), self.init_c(img_mean)


class EncoderDecoder(nn.Module):
    def __init__(self, model_name, pretrained=False, vocab_size=100, dropout=0.1,
                 output_size=(14, 14), embed_dim=512, decoder_dim=512, attn_dim=512):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(model_name, pretrained, output_size)
        self.decoder = Decoder(256, embed_dim, decoder_dim, attn_dim, vocab_size, dropout)

    def forward_impl(self, img, text, text_len):
        img = self.encoder(img)
        return self.decoder(img, text, text_len)

    def forward(self, img, text, text_len):
        return self.forward_impl(img, text, text_len)


def get_model(*args, **kwargs):
    return EncoderDecoder(*args, **kwargs)

