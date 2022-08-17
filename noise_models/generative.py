import json
import os
import pickle
import sys

from PIL import Image
import torch
import transformers
import tqdm


# get access to torch_utils and dnnlib from stylegan
stylegan_path = os.path.expanduser('~/code/stylegan2-ada-pytorch')
sys.path.append(stylegan_path)

pretrained_stylegan_path = os.path.expanduser(
    '~/store/remote_logs/manel/stylegan2/shaders-21k-best-gamma/'
    '00000-shaders21k_256x256-auto4-gamma5-bgcfnc/network-snapshot-025000.pkl')

coco_captions_path = os.path.expanduser('~/store/datasets/coco/annotations/captions_train2017.json')

save_dir = os.path.expanduser('~/store/datasets/generative/coco-to-shaders-stylegan')
os.makedirs(save_dir, exist_ok=True)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class COCOCaptions(torch.utils.data.Dataset):
    def __init__(self, anno_file):
        self.anno_file = anno_file
        j = json.load(open(anno_file))
        self.captions = [x['caption'] for x in j['annotations']]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


@torch.no_grad()
def generate_images_from_captions(limit=-1, batch_size=64):
    caption_text = COCOCaptions(coco_captions_path).captions

    # load pre-trained stylegan
    with open(pretrained_stylegan_path, 'rb') as f:
        generator = pickle.load(f)['G_ema'].to(device).eval()
    z_dim = generator.z_dim

    # load pre-trained BERT
    # config = transformers.AutoConfig('bert-base-uncased')
    # text_encoder = transformers.AutoModel.from_config(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    text_encoder = transformers.AutoModel.from_pretrained('bert-base-uncased').to(device).eval()
    embed_size = text_encoder.config.hidden_size

    rgen = torch.Generator(device=device).manual_seed(0xAAAAAAAA)
    projection_matrix = torch.randn(embed_size, z_dim, generator=rgen, device=device)
    c = torch.zeros(1, generator.c_dim, device=device)
    
    limit = limit if limit > 0 else len(caption_text)
    for i in tqdm.trange(0, limit, batch_size):
        # extract text embedding
        text = caption_text[i:i + batch_size]
        text = tokenizer(text, padding='longest', truncation=True, max_length=128)
        att_mask = torch.as_tensor(text.attention_mask, device=device)
        text = torch.as_tensor(text.input_ids, device=device)
        text_out = text_encoder(text, attention_mask=att_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        # map to latent space and normalize
        z = text_feat @ projection_matrix
        z.div_(z.std())
        w = generator.mapping(z, c=None, truncation_psi=1, truncation_cutoff=None)
        # synthesize images
        imgs = generator.synthesis(w, noise_mode='const')
        imgs = imgs.mul_(127.5).add_(128).clamp_(0, 255).byte().permute(0, 2, 3, 1)
        for k in range(imgs.shape[0]):
            img = Image.fromarray(imgs[k].cpu().numpy())
            img.save(os.path.join(save_dir, f'{i+k:>06d}.jpg'))

if __name__ == '__main__':
    generate_images_from_captions(limit=512)
