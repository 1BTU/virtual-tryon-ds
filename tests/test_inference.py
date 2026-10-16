import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image
from torchvision import transforms

from inference_dataset import VITONInferenceDataset
from never_parse_ds_test_SDAFNet_viton import atest, get_opt, load_model, resolve_device


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super(RecordingModel, self).__init__()
        self.norm = torch.nn.BatchNorm2d(3)
        self.calls = []

    def forward(self, person, cloth):
        self.calls.append((tuple(person.shape), self.training, torch.is_grad_enabled()))
        return self.norm(person) * 0 + cloth


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'test_img').mkdir()
        (self.root / 'clothes').mkdir()
        # A partial final batch must still produce a file for every pair.
        for i in range(3):
            Image.new('RGB', (300, 400), (20 + i, 40, 60)).save(
                str(self.root / 'test_img' / '{}_0.jpg'.format(i)))
        Image.new('RGB', (300, 400), (180, 90, 45)).save(
            str(self.root / 'clothes' / 'shirt_1.jpg'))
        self.pairs = self.root / 'pairs.txt'
        self.pairs.write_text('\n'.join('{}_0.jpg shirt_0.jpg'.format(i) for i in range(3)))
        self.opt = get_opt(['--dataset_imgpath', str(self.root),
                            '--dataset_list', str(self.pairs), '--save_dir', str(self.root / 'out'),
                            '--device', 'cpu', '-j', '0', '-b', '2'])

    def test_preprocessing_matches_original_and_reads_only_two_images(self):
        dataset = VITONInferenceDataset(self.opt)
        with mock.patch('inference_dataset.Image.open', wraps=Image.open) as opened:
            sample = dataset[0]
        self.assertEqual(opened.call_count, 2)
        original = transforms.Compose([
            transforms.Resize(192, interpolation=2), transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        with Image.open(str(self.root / 'clothes' / 'shirt_1.jpg')) as image:
            self.assertTrue(torch.equal(sample['cloth'], original(image)))
        self.assertEqual(tuple(sample['img'].shape), (3, 256, 192))

    def test_cpu_batches_eval_mode_normalization_and_comparison(self):
        model = RecordingModel().train()
        output = atest(self.opt, model)
        self.assertEqual(model.calls, [((2, 3, 256, 192), False, False),
                                       ((1, 3, 256, 192), False, False)])
        self.assertEqual(len(list(output.glob('*.jpg'))), 3)
        with Image.open(str(output / '0_0.jpg')) as image:
            self.assertEqual(image.size, (192, 256))
            with Image.open(str(self.root / 'clothes' / 'shirt_1.jpg')) as source:
                for got, expected in zip(image.getpixel((50, 50)), source.getpixel((50, 50))):
                    self.assertLessEqual(abs(got - expected), 3)
        self.opt.add_compare = True
        atest(self.opt, model)
        with Image.open(str(output / '0_0.jpg')) as image:
            self.assertEqual(image.size, (576, 256))

    def test_nonstandard_image_size_is_resized_for_model(self):
        Image.new('RGB', (200, 200)).save(str(self.root / 'test_img' / '0_0.jpg'))
        self.opt.batch_size = 1
        model = RecordingModel()
        atest(self.opt, model)
        self.assertEqual(model.calls[0][0], (1, 3, 256, 192))

    def test_worker_processes_preserve_all_outputs(self):
        self.opt.workers = 2
        model = RecordingModel()
        output = atest(self.opt, model)
        self.assertEqual(len(list(output.glob('*.jpg'))), 3)
        self.assertEqual(len(model.calls), 2)

    def test_full_model_checkpoint_and_invalid_checkpoints(self):
        path = str(self.root / 'model.pt')
        torch.save(RecordingModel(), path)
        model = load_model(path, torch.device('cpu'))
        self.assertFalse(model.training)
        self.assertEqual(next(model.parameters()).device.type, 'cpu')
        torch.save(model.state_dict(), path)
        with self.assertRaisesRegex(TypeError, 'state_dict'):
            load_model(path, torch.device('cpu'))
        with self.assertRaises(FileNotFoundError):
            load_model(str(self.root / 'missing.pt'), torch.device('cpu'))

    def test_clear_pair_list_errors(self):
        for contents in ('', 'only_one.jpg', '../escape.jpg shirt_1.jpg'):
            self.pairs.write_text(contents)
            with self.assertRaises(ValueError):
                VITONInferenceDataset(self.opt)

    def test_cuda_unavailable_is_reported(self):
        with mock.patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto').type, 'cpu')
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                resolve_device('cuda')

    def test_student_model_import_does_not_require_visualization_packages(self):
        from models.never_parse_ds_studentmodel import NPDNet
        self.assertTrue(issubclass(NPDNet, torch.nn.Module))


if __name__ == '__main__':
    unittest.main()
