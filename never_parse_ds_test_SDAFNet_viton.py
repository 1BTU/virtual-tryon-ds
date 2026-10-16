"""Batch inference for trusted, serialized parser-free student models."""

import argparse
import inspect
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision.utils import save_image
import tqdm

from inference_dataset import VITONInferenceDataset


def get_opt(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='TEST_PAIR')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=192)
    parser.add_argument('--mode', default='test')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--add_compare', action='store_true')
    parser.add_argument('--dataset_imgpath', default='data/VITON/VITON_test')
    parser.add_argument('--dataset_list', default='data/VITON/test_any.txt')
    parser.add_argument('--checkpoint_dir', default=(
        'checkpoint/pf_b_2l2e-05m_35/pf_b_2l2e-05m_35_data.pt'))
    parser.add_argument('--save_dir', default='./results/')
    parser.add_argument('--multi_flows', type=int, default=6)
    parser.add_argument('--display_freq', type=int, default=100)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    opt = parser.parse_args(argv)
    if opt.batch_size < 1 or opt.workers < 0:
        parser.error('batch_size must be positive and workers must be nonnegative')
    if (opt.load_height, opt.load_width) != (256, 192):
        parser.error('the supplied student models require load_height=256, load_width=192')
    return opt


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable; use --device cpu')
    return torch.device(name)


def load_model(checkpoint, device):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError('Checkpoint not found: {}. See README.md for weights.'.format(checkpoint))
    kwargs = {'map_location': 'cpu'}
    # Newer PyTorch defaults to weights-only loading; legacy checkpoints store
    # nn.Module objects. Only load checkpoints you trust.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    model = torch.load(checkpoint, **kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError('Expected a serialized nn.Module; a state_dict needs its matching model definition')
    return model.to(device).eval()


def atest(opt, net):
    device = resolve_device(opt.device)
    net = net.to(device).eval()
    output_dir = Path(opt.save_dir) / opt.name / 'vis_viton_out'
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = VITONInferenceDataset(opt)
    # CUDA page-locked batches permit asynchronous host-to-device copies.
    loader = data.DataLoader(dataset, batch_size=opt.batch_size,
                             shuffle=opt.shuffle, num_workers=opt.workers,
                             pin_memory=device.type == 'cuda')
    with torch.no_grad():
        for inputs in tqdm.tqdm(loader):
            img = inputs['img'].to(device, non_blocking=device.type == 'cuda')
            cloth = inputs['cloth'].to(device, non_blocking=device.type == 'cuda')
            size = (256, 192)
            if img.shape[-2:] != size:
                img = F.interpolate(img, size=size, mode='bilinear', align_corners=False)
            if cloth.shape[-2:] != size:
                cloth = F.interpolate(cloth, size=size, mode='bilinear', align_corners=False)
            # Preserve the argument order used by the student training script.
            result = net(img, cloth)
            if opt.add_compare:
                result = torch.cat((img, cloth, result), dim=3)
            # Transfer once per batch, rather than synchronizing for every file.
            result = result.cpu()
            for name, image in zip(inputs['img_name'], result):
                # Explicit normalization works across torchvision versions.
                save_image(image.add(1).div(2).clamp(0, 1), str(output_dir / name))
    return output_dir


def main():
    opt = get_opt()
    device = resolve_device(opt.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model = load_model(opt.checkpoint_dir, device)
    output_dir = atest(opt, model)
    print('Results saved to {}'.format(output_dir))


if __name__ == '__main__':
    main()
