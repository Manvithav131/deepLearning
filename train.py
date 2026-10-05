import os
import re
import time
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, avoids popup windows in scripts
import matplotlib.pyplot as plt

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms, models


# ---------------- Text preprocessing ----------------
def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^a-z\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption


def caption_to_indices(caption, word2idx):
    tokens = ['<start>'] + caption.split() + ['<end>']
    return [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]


# ---------------- Dataset ----------------
class FlickrDataset(Dataset):
    def __init__(self, df, images_dir, word2idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.word2idx = word2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.images_dir, row['image'])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption_tensor = torch.tensor(caption_to_indices(row['caption_clean'], self.word2idx), dtype=torch.long)
        return image, caption_tensor


class CollateFn:
    """Class-based collate function so it can be pickled by Windows multiprocessing."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        captions_padded = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions_padded


# ---------------- Encoder (spatial feature map for attention) ----------------
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=7, fine_tune=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]  # drop avgpool + fc, keep spatial conv output
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        for p in self.resnet.parameters():
            p.requires_grad = False
        if fine_tune:
            # unfreeze last residual block (layer4) only
            for p in list(self.resnet.children())[-1].parameters():
                p.requires_grad = True

    def forward(self, images):
        out = self.resnet(images)          # [B, 512, H, W]
        out = self.adaptive_pool(out)      # [B, 512, 7, 7]
        out = out.permute(0, 2, 3, 1)      # [B, 7, 7, 512]
        return out


# ---------------- Attention module ----------------
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)                          # [B, num_pixels, attn_dim]
        att2 = self.decoder_att(decoder_hidden)                       # [B, attn_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [B, num_pixels]
        alpha = self.softmax(att)                                     # [B, num_pixels]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)       # [B, encoder_dim]
        return context, alpha


# ---------------- Decoder with attention ----------------
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512):
        super().__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        return self.init_h(mean_encoder_out), self.init_c(mean_encoder_out)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # [B, 49, 512]

        embeddings = self.embedding(captions[:, :-1])                 # feed all but last token
        h, c = self.init_hidden_state(encoder_out)

        seq_len = embeddings.size(1)
        predictions = torch.zeros(batch_size, seq_len, self.fc.out_features, device=encoder_out.device)

        for t in range(seq_len):
            context, _ = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            predictions[:, t, :] = self.fc(h)

        return predictions


# ---------------- Inference (greedy, attention-aware) ----------------
def generate_caption(image_tensor, encoder, decoder, word2idx, idx2word, device, max_len=20):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        encoder_out = encoder(image_tensor)
        encoder_out = encoder_out.view(1, -1, encoder_out.size(-1))

        h, c = decoder.init_hidden_state(encoder_out)
        word = torch.tensor([word2idx['<start>']], device=device)
        result = []

        for _ in range(max_len):
            embed = decoder.embedding(word)
            context, _ = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
            h, c = decoder.decode_step(torch.cat([embed, context], dim=1), (h, c))
            preds = decoder.fc(h)
            predicted = preds.argmax(dim=1)

            if predicted.item() == word2idx['<end>']:
                break
            result.append(predicted.item())
            word = predicted

    return ' '.join(idx2word[i] for i in result if idx2word[i] not in ['<start>', '<pad>'])


# ---------------- Main ----------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)

    # --- confirmed local paths ---
    base = r'C:\Users\User\OneDrive\Desktop\DeepLearning\archive'
    images_dir = os.path.join(base, 'Images')
    captions_path = os.path.join(base, 'captions.txt')
    checkpoint_dir = r'.\checkpoints'
    output_dir = r'.\outputs'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- data ---
    captions_df = pd.read_csv(captions_path)
    captions_df['caption_clean'] = captions_df['caption'].apply(clean_caption)

    counter = Counter()
    for cap in captions_df['caption_clean']:
        counter.update(cap.split())
    MIN_FREQ = 5
    vocab_words = [w for w, f in counter.items() if f >= MIN_FREQ]
    vocab = ['<pad>', '<start>', '<end>', '<unk>'] + sorted(vocab_words)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    print("Vocab size:", len(vocab))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FlickrDataset(captions_df, images_dir, word2idx, transform=transform)

    BATCH_SIZE = 16  # reduce to 8 if you hit CUDA out-of-memory on the 6GB GPU
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=CollateFn(word2idx['<pad>']),
        num_workers=4
    )

    # --- model ---
    EMBED_SIZE = 256
    ATTENTION_DIM = 256
    DECODER_DIM = 512
    VOCAB_SIZE = len(vocab)
    NUM_EPOCHS = 20

    encoder = EncoderCNN(fine_tune=True).to(device)
    decoder = DecoderWithAttention(ATTENTION_DIM, EMBED_SIZE, DECODER_DIM, VOCAB_SIZE).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = optim.Adam([
        {'params': decoder.parameters(), 'lr': 1e-3},
        {'params': [p for p in encoder.parameters() if p.requires_grad], 'lr': 1e-5},
    ])

    # --- training loop ---
    encoder.train()
    decoder.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        start_time = time.time()
        for batch_idx, (images, captions) in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)

            encoder_out = encoder(images)
            outputs = decoder(encoder_out, captions)   # [B, seq_len-1, vocab_size]
            targets = captions[:, 1:]                   # shifted by one (predicting next word)

            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"===> Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s\n")

        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'word2idx': word2idx,
            'idx2word': idx2word,
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth'))

    # --- qualitative check: save sample images with captions as PNGs ---
    for i in [0, 100, 5000]:
        img_tensor, _ = dataset[i]
        cap = generate_caption(img_tensor, encoder, decoder, word2idx, idx2word, device)
        print(f"Image {i} generated caption: {cap}")

        img_display = img_tensor.permute(1, 2, 0).numpy()
        img_display = img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_display = img_display.clip(0, 1)

        plt.figure()
        plt.imshow(img_display)
        plt.axis('off')
        plt.title(cap, fontsize=10)
        save_path = os.path.join(output_dir, f'sample_{i}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()