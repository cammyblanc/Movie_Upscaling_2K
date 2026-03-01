import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# --- 修正ポイント: 全スキャンをやめ、必要な CodeFormer だけを明示的に読み込む ---
# これにより、arcface_arch などの不足しているモジュールによるエラーを回避します
try:
    _arch_modules = [importlib.import_module('codeformer.codeformer_arch')]
except ImportError:
    # 以前の階層構造の場合のフォールバック
    _arch_modules = [importlib.import_module('codeformer.archs.codeformer_arch')]

def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net