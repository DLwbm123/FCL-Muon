import os
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from research.continual_muon.current_cifar import CurrentCIFAR, image_payload_offset


class CurrentCIFARTest(unittest.TestCase):
    def test_indexed_read_and_reject_other_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'train')
            header = (b'\x80\x02' + b'N' * 120 +
                      b'K\x01MP\xc3M\x00\x0c\x86cnumpy\ndtype\nU\x02u1\x89T' +
                      struct.pack('<I', 50000 * 3072))
            with open(path, 'wb') as f:
                f.write(header)
                offset = f.tell()
                f.seek(offset + 7 * 3072)
                f.write(bytes(range(256)) * 12)
                f.seek(offset + 50000 * 3072)
                f.write(b'tbu.')
            self.assertEqual(image_payload_offset(path), offset)
            data = CurrentCIFAR(path, list(range(50000)), {1: [7], 2: [8]})
            try:
                with patch('os.pread', wraps=os.pread) as read:
                    image, label = data[7]
                    read.assert_called_once_with(data.fd, 3072, offset + 7 * 3072)
                expected = np.frombuffer(bytes(range(256)) * 12, np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
                np.testing.assert_array_equal(np.asarray(image), expected)
                from torchvision.datasets import CIFAR100
                reference = CIFAR100.__new__(CIFAR100)
                reference.data = np.zeros((8, 32, 32, 3), dtype=np.uint8)
                reference.data[7] = expected
                reference.targets = list(range(8))
                reference.transform = reference.target_transform = None
                torchvision_image, torchvision_label = reference[7]
                np.testing.assert_array_equal(np.asarray(image), np.asarray(torchvision_image))
                self.assertEqual(label, torchvision_label)
                self.assertEqual(label, 7)
                with self.assertRaises(ValueError):
                    data[8]
                data.enter_t3()
                with self.assertRaises(ValueError):
                    data[7]
                self.assertEqual(data.reads, {1: 1, 2: 0})
            finally:
                data.close()


if __name__ == '__main__':
    unittest.main()
