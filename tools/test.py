import argparse
import logging

import cv2
import torch
import sklearn
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.fusionad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pickle

warnings.filterwarnings("ignore")

activations = {} 

def tensor_to_heatmap(feature_map: torch.Tensor) -> np.ndarray:
    """
    feature_map: 形状が (C, H, W) のTensor (1枚のカメラ分)を想定。
    戻り値: ヒートマップ（カラー）のBGR画像 (H, W, 3)
    """
    # ---- 1. チャネル方向を平均して1枚にする (H, W)
    # 必要に応じて mean/max/sum などを切り替える
    heatmap = feature_map.mean(dim=0)  # -> shape: (H, W)

    # ---- 2. テンソル → numpy & 正規化
    heatmap = heatmap.detach().cpu().numpy()
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)

    # ---- 3. カラーマップを適用 (OpenCVはBGR)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def overlay_heatmap_on_image(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    image_bgr: 元画像 (H, W, 3)
    heatmap_bgr: ヒートマップ (H, W, 3)
    alpha: ヒートマップ重ねる割合
    戻り値: ヒートマップを重ね合わせたBGR画像
    """
    # サイズが違う場合は合わせる
    if (image_bgr.shape[0] != heatmap_bgr.shape[0]) or (image_bgr.shape[1] != heatmap_bgr.shape[1]):
        heatmap_bgr = cv2.resize(heatmap_bgr, (image_bgr.shape[1], image_bgr.shape[0]))

    # addWeighted でオーバーレイ
    overlaid = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlaid


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--result_file', type=str, default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    print(args.config)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                logging.info(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                logging.info(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    def get_activation_hook(name):
        def hook(module, input, output):
            print(f"[HOOK CALLED] {name}")
            if name not in activations:
                activations[name] = [] 

            # module: フックを登録した層 (nn.Module)
            # input:  その層に入ってきた入力 (tuple of Tensor)
            # output: その層から出力される出力 (Tensor or tuple of Tensor)

            x = input[0] if isinstance(input, (list, tuple)) else input
            if isinstance(x, torch.Tensor):
                print(f"[{name}] input shape: {x.shape}")
            else:
                print(f"[{name}] input is a {type(x)}")

            # 出力に対して
            if isinstance(output, torch.Tensor):
                # 通常のTensorの場合
                print(f"[{name}] output shape: {output.shape}")
            elif isinstance(output, (list, tuple)):
                # タプルやリストなら要素ごとに
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        print(f"[{name}] output[{i}] shape: {out.shape}")
                    else:
                        print(f"[{name}] output[{i}] is a {type(out)}")
            else:
                print(f"[{name}] output is a {type(output)}")
            
            activations[name].append(output.detach().cpu())
            print(f'activation shape: {output.shape}')
            print(f'activation shape: {activations[name][-1].shape}')
        return hook

    print('--- Named Modules ---')
    count = 0
    for name, module in model.named_modules():
        if 'img_backbone.layer4' in name:
            print(name, "->", module.__class__.__name__)
            count += 1

    for name, module in model.named_modules():
        if 'img_neck.fpn' in name:
            print(name, "->", module.__class__.__name__)

    print(f"Total modules containing 'img_backbone.layer4': {count}")

    # for name, module in model.named_modules():
    #     print(name, module)

    valid_names = [
        'img_backbone.layer4.2'
    ]

    for name, module in model.named_modules():
        # print(f'Registering hook for {name}')
        if name in valid_names:
            print(f'Registering hook for {name}')
            module.register_forward_hook(get_activation_hook(name))

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

 
    if args.result_file == None: 
        if not distributed:
            assert False
            # model = MMDataParallel(model, device_ids=[0])
            # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)
   
    rank, _ = get_dist_info()

    pkl_path = '/home/yoshi-22/FusionAD/UniAD/data/infos/nuscenes_infos_temporal_val.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    infos = data['infos']

    cam_out_path = [
        '/home/yoshi-22/FusionAD/outputs/CAM_FRONT',
        '/home/yoshi-22/FusionAD/outputs/CAM_FRONT_RIGHT',
        '/home/yoshi-22/FusionAD/outputs/CAM_FRONT_LEFT',
        '/home/yoshi-22/FusionAD/outputs/CAM_BACK',
        '/home/yoshi-22/FusionAD/outputs/CAM_BACK_LEFT',
        '/home/yoshi-22/FusionAD/outputs/CAM_BACK_RIGHT'
    ]

    cam_names = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]

    name = 'img_backbone.layer4.2'
    my_activations = outputs['my_activations']

    for i in sorted(my_activations[name].keys()):

        feat_i = my_activations[name][i]
        print(f"Processing frame {i}")
        info_i = infos[i]

        for cam_idx in range(6):
            single_feat_map = feat_i[cam_idx]
            # カメラごとのファイル名に cam_names[cam_idx] を入れる
            heatmap_bgr = tensor_to_heatmap(single_feat_map)

            # 入力画像のパス
            img_path = info_i['cams'][cam_names[cam_idx]]['data_path']
            print(f"Reading image: {img_path}")
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"Failed to read image: {img_path}")
                continue

            # feat[cam_idx] → (C, H, W)
            overlaid = overlay_heatmap_on_image(image_bgr, heatmap_bgr, alpha=0.5)

            # 出力ファイル名
            out_name = os.path.basename(img_path)
            # ディレクトリが存在するかを事前にチェック
            if not os.path.exists(cam_out_path[cam_idx]):
                os.makedirs(cam_out_path[cam_idx], exist_ok=True)
                print(f"Directory created: {cam_out_path[cam_idx]}")
            else:
                print(f"Directory already exists: {cam_out_path[cam_idx]}")
            out_path = os.path.join(cam_out_path[cam_idx], out_name)
            cv2.imwrite(out_path, overlaid)
            print(f"Heatmap saved to {out_path}")


    if rank == 0:
        if args.result_file != None:
            outputs = mmcv.load(args.out)
        elif args.out:
            logging.info(f'\nwriting results to {args.out}')
            #assert False
            mmcv.dump(outputs, args.out)
            #outputs = mmcv.load(args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            print('formating results......')
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print('evaluating results......')
            logging.info(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
