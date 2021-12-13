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
                 output_size=(14, 14), embed_dim=512, decoder_dim=512, attn_dim=512, beam=5):
        super(EncoderDecoder, self).__init__()
        self.beam = beam
        self.vocab_size = vocab_size
        self.encoder = Encoder(model_name, pretrained, output_size)
        self.decoder = Decoder(256, embed_dim, decoder_dim, attn_dim, vocab_size, dropout)

    def forward_impl(self, img, text, text_len):
        img = self.encoder(img)
        return self.decoder(img, text, text_len)

    def forward(self, img, text, text_len):
        return self.forward_impl(img, text, text_len)

    def inference(self, img, word2idx, idx2word):
        """Apply beam search (batch_size = 1)"""
        beam, device = self.beam, img.device
        img = rearrange(self.encoder(img.unsqueeze(0)), '1 h w c -> 1 (h w) c').repeat(beam, 1, 1)

        previous_words = torch.LongTensor([[word2idx['<start>']]] * beam).to(device)
        sentences, sentence_scores = previous_words, torch.zeros(beam, 1).to(device)
        complete_sentences, complete_sentence_scores = list(), list()

        step = 1

        h, c = self.decoder.init_h_c(img)

        while beam > 0 and step <= 50:
            text = self.decoder.embed(previous_words).squeeze(1)
            attn_weighted_img, attn = self.decoder.attn(img, h)
            attn_weighted_img *= self.decoder.gate(h)
            h, c = self.decoder.decoder(torch.cat([attn_weighted_img, text], dim=1), (h, c))
            scores = sentence_scores + F.log_softmax(self.decoder.head(h), dim=1)

            top_k_scores, top_k_words = scores.view(-1).topk(beam, 0)
            prev_sentence_idx = top_k_words // self.vocab_size
            next_word_idx = top_k_words % self.vocab_size
            sentences = torch.cat([sentences[prev_sentence_idx], next_word_idx.unsqueeze(1)], dim=1)

            incomplete_idx = [idx for idx, next_word in enumerate(next_word_idx) if next_word != word2idx['<end>']]
            complete_idx = list(set(range(len(next_word_idx))) - set(incomplete_idx))
            org_sentence_idx = prev_sentence_idx[incomplete_idx]

            if len(complete_idx) > 0:
                complete_sentences.extend(sentences[complete_idx].tolist())
                complete_sentence_scores.extend(top_k_scores[complete_idx])
                beam -= len(complete_idx)

            sentences, sentence_scores = sentences[incomplete_idx], sentence_scores[incomplete_idx]
            h, c, img = h[org_sentence_idx], c[org_sentence_idx], img[org_sentence_idx]
            previous_words = next_word_idx[incomplete_idx]

            step += 1

        max_idx = complete_sentence_scores.index(max(complete_sentence_scores))

        sentence = " ".join([idx2word[w] for w in complete_sentences[max_idx]
                if w not in {word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']}])

        return sentence


def get_model(*args, **kwargs):
    return EncoderDecoder(*args, **kwargs)

