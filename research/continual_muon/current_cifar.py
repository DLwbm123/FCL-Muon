"""Read only requested CIFAR-100 train image records from the original pickle."""
import os
import pickletools
import struct
import numpy as np
from PIL import Image


def image_payload_offset(path):
    # Parse pickle opcodes without loading string payloads; no unpickling or image read.
    opcodes = {ord(op.code): op for op in pickletools.opcodes}
    with open(path, 'rb', buffering=0) as f:
        if f.read(2) != b'\x80\x02':
            raise ValueError('unexpected pickle protocol')
        f.seek(0)
        while True:
            code = f.read(1)
            if not code or code[0] not in opcodes:
                raise ValueError('invalid CIFAR pickle')
            op = opcodes[code[0]]
            if op.name in ('BINSTRING', 'BINBYTES', 'BINUNICODE'):
                length = struct.unpack('<I', f.read(4))[0]
                offset = f.tell()
                if length >= 10_000_000:
                    if op.name != 'BINSTRING' or length != 50_000 * 3072:
                        raise ValueError('unexpected large pickle payload')
                    f.seek(offset - 100)
                    header = f.read(100)
                    if b'K\x01MP\xc3M\x00\x0c\x86cnumpy\ndtype\n' not in header or b'U\x02u1' not in header:
                        raise ValueError('unexpected image array shape or dtype')
                    f.seek(offset + length)
                    if f.read(4) != b'tbu.':
                        raise ValueError('unexpected image payload trailer')
                    return offset
                f.seek(length, os.SEEK_CUR)
            elif op.name in ('SHORT_BINSTRING', 'SHORT_BINBYTES'):
                size = f.read(1)
                if not size:
                    raise ValueError('truncated pickle')
                f.seek(size[0], os.SEEK_CUR)
            elif op.name in ('BINBYTES8', 'BINUNICODE8', 'BYTEARRAY8'):
                raise ValueError('unexpected pickle payload type')
            elif op.arg is not None:
                op.arg.reader(f)
            if op.name == 'STOP':
                raise ValueError('missing image payload')


class CurrentCIFAR:
    def __init__(self, path, labels, task_indices):
        self.offset = image_payload_offset(path)
        self.fd = os.open(path, os.O_RDONLY)
        self.targets = list(labels)
        if len(self.targets) != 50_000:
            os.close(self.fd)
            raise ValueError('unexpected train label count')
        self.allowed = {k: frozenset(v) for k, v in task_indices.items()}
        if set(self.allowed) != {1, 2} or self.allowed[1] & self.allowed[2]:
            os.close(self.fd)
            raise ValueError('expected disjoint T2/T3 partitions')
        self.task = 1
        self.reads = {1: 0, 2: 0}

    def __getitem__(self, index):
        index = int(index)
        if index not in self.allowed[self.task] or not 0 <= index < 50_000:
            raise ValueError('only current-task train images may be read')
        data = os.pread(self.fd, 3072, self.offset + index * 3072)
        if len(data) != 3072:
            raise ValueError('truncated image record')
        self.reads[self.task] += 1
        image = np.frombuffer(data, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
        return Image.fromarray(image), int(self.targets[index])

    def enter_t3(self):
        if self.task != 1:
            raise ValueError('invalid or repeated task switch')
        self.task = 2

    def close(self):
        os.close(self.fd)
