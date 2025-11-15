import os.path as osp
import time
import torch
from config import config
from multireg.engine import Engine
from multireg.utils.torch_utils import to_cuda

from model import create_model
from dataset import ROBI_test_data_loader
from loss import Evaluator
from multireg.utils.point_cloud_utils import  apply_transform

try:
	from pointscope import PointScopeClient as PSC
	# from pointscope import PointScopeVedo as PSC
except ImportError:
	pass

eps = 1e-8

def visualization(data_dict, output_dict, metrics):
    src_points_f = output_dict["src_points_f"]
    ref_points_f = output_dict["ref_points_f"]
    estimated_transform = output_dict["estimated_transforms"]
    all_ref_corr_points = output_dict['all_ref_corr_points']
    all_src_corr_points = output_dict['all_src_corr_points']
    psc = PSC().vedo(bg_color=[1.0, 1.0, 1.0], subplot=4)\
        .add_pcd(src_points_f, estimated_transform[0]) \
        .draw_at(1) \
        .add_pcd(ref_points_f) \
        .draw_at(2)
    psc.add_pcd(src_points_f).add_pcd(ref_points_f).add_lines(all_src_corr_points, all_ref_corr_points, color=[1,0,0]).draw_at(3)
    psc.add_pcd(ref_points_f)
    for i in range(len(estimated_transform)):
        psc.add_pcd(src_points_f, estimated_transform[i])
    psc.show()

def run_one_epoch(
        engine,
        data_loader,
        model,
        evaluator,
        training=True
):
    if training:
        model.train()
    else:
        model.eval()
    inlier_ratio_num=0
    mean_pre=0
    mean_recall=0
    for i, data_dict in enumerate(data_loader):
        data_dict = to_cuda(data_dict)
        with torch.no_grad():
            output_dict = model(data_dict)
            estimated_transforms_gt=data_dict['transform']
            all_ref_corr_points = output_dict['all_ref_corr_points']
            all_src_corr_points = output_dict['all_src_corr_points']
            metrics = evaluator(output_dict, data_dict)
            mean_pre += metrics['precision']
            mean_recall += metrics['recall']
            corr_tensor = torch.cat((all_src_corr_points, all_ref_corr_points), dim=1)
            align_src_points = apply_transform(corr_tensor[:,:3].unsqueeze(0), estimated_transforms_gt)
            rmse = torch.linalg.norm(align_src_points - corr_tensor[:,3:].unsqueeze(0), dim=-1)<(0.005)
            inlier_ratio = (rmse.float().sum(0)>0)
            inlier_ratio = inlier_ratio.float().sum()/len(inlier_ratio)
            metrics['inlier_ratio'] = inlier_ratio
            msg = 'Iter [{}/{}]: precision: {:.3f}, '.format(
                i + 1,
                len(data_loader),
                100*mean_pre/(i+1)) + \
                'recall: {:.3f}, '.format(100*mean_recall/(i+1)) + \
                'inlier_ratio: {:.3f}. '.format(100*inlier_ratio)
            engine.logger.info(msg)

            if len(all_src_corr_points)>1:
                inlier_ratio_num+=inlier_ratio
            visualization(data_dict, output_dict, metrics)

    recall=100*mean_recall/len(data_loader)
    precision=100*mean_pre/len(data_loader)

    message = 'precision: {:.3f}, '.format(precision) + \
                'recall: {:.3f}, '.format(recall) + \
                'inlier_ratio: {:.3f}, '.format(100*inlier_ratio_num/(len(data_loader))) + \
                'f1: {:.3f}. '.format(2 * (precision ) * (recall) / ((recall  +precision)))
    engine.logger.info(message)


def main():
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, seed=config.seed) as engine:
        start_time = time.time()

        test_loader,neighborhood_limits = ROBI_test_data_loader(engine, config)
        loading_time = time.time() - start_time

        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config).cuda()
        evaluator = Evaluator(config).cuda()

        engine.register_state(model=model)
        engine.load_snapshot(engine.args.snapshot)

        start_time = time.time()
        run_one_epoch(engine, test_loader, model, evaluator, training=False)

        loading_time = time.time() - start_time
        message = ' test_one_epoch: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)


if __name__ == '__main__':
    main()
