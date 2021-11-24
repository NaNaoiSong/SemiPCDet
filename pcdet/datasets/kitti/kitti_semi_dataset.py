import copy
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from . import kitti_utils
from ..semi_dataset import SemiDatasetTemplate
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti

def split_kitti_semi_data(info_paths, data_splits, root_path, labeled_ratio, logger, data_cfg, save_dir):
    assert 0 < labeled_ratio < 1, 'labeled_ratio value error'
    kitti_pretrain_infos = []
    kitti_test_infos = []
    kitti_labeled_infos = []
    kitti_unlabeled_infos = []

    root_path = Path(root_path)
    labeled_split = []

    train_split = data_splits['train']
    exist_check = False
    for info_path in info_paths[train_split]:
        info_path = root_path / info_path
        split_txt_name = '{}_{}.txt'.format(info_path.stem.split('_')[-1], labeled_ratio)
        save_path = save_dir / split_txt_name
        exist_check = save_path.exists()
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        num_infos = len(infos)
        if exist_check:
            labeled_inds = [int(x.strip().split(' ')[-1]) for x in open(str(save_path)).readlines()]
        else:
            labeled_inds = sorted(np.random.choice(num_infos, int(num_infos * labeled_ratio), replace=False).tolist())
        unlabeled_inds = list(set(range(num_infos)) - set(labeled_inds))
        labeled_infos = [infos[i] for i in labeled_inds]
        unlabeled_infos = [infos[i] for i in unlabeled_inds]
        cur_split = [info['point_cloud']['lidar_idx'] for info in labeled_infos]
        labeled_split.extend(cur_split)
        if not exist_check:
            output_split = ['{} {}\n'.format(s, i) for s, i in zip(cur_split, labeled_inds)]
            with open(str(save_dir / split_txt_name), 'w') as f:
                f.writelines(output_split)

        kitti_pretrain_infos.extend(copy.deepcopy(labeled_infos))
        kitti_labeled_infos.extend(copy.deepcopy(labeled_infos))
        kitti_unlabeled_infos.extend(copy.deepcopy(unlabeled_infos))

    # generate part db_infos
    aug_list = [aug.NAME for aug in data_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST]
    labeled_split = set(labeled_split)
    if 'gt_sampling' in aug_list:
        idx = aug_list.index('gt_sampling')
        db_paths = data_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[idx].DB_INFO_PATH
        new_db_paths = []
        for db_path in db_paths:
            db_path = root_path / db_path
            if not exist_check:
                with open(str(db_path), 'rb') as f:
                    infos = pickle.load(f)
                new_db_infos = {}
                for key in infos.keys():
                    new_db_infos[key] = [info for info in infos[key] if info['image_idx'] in labeled_split]
                new_path = str(save_dir / Path(db_path).stem) + '_part.pkl'
                with open(new_path, 'wb') as f:
                    pickle.dump(new_db_infos, f)
            else:
                new_path = str(save_dir / Path(db_path).stem) + '_part.pkl'
            new_db_paths.append(new_path)
        data_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[idx].DB_INFO_PATH = new_db_paths

    test_split = data_splits['test']
    for info_path in info_paths[test_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            kitti_test_infos.extend(copy.deepcopy(infos))

    logger.info('Total samples for KITTI pre-training dataset: %d' % (len(kitti_pretrain_infos)))
    logger.info('Total samples for KITTI testing dataset: %d' % (len(kitti_test_infos)))
    logger.info('Total samples for KITTI labeled dataset: %d' % (len(kitti_labeled_infos)))
    logger.info('Total samples for KITTI unlabeled dataset: %d' % (len(kitti_unlabeled_infos)))

    return kitti_pretrain_infos, kitti_test_infos, kitti_labeled_infos, kitti_unlabeled_infos


def split_kitti_semi_test_data(info_paths, data_splits, root_path, labeled_ratio, logger):
    kitti_test_infos = []
    root_path = Path(root_path)
    test_split = data_splits['test']
    for info_path in info_paths[test_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            kitti_test_infos.extend(copy.deepcopy(infos))

    logger.info('Total samples for KITTI testing dataset: %d' % (len(kitti_test_infos)))
    return kitti_test_infos


class KITTISemiDataset(SemiDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = infos

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict


class KITTIPretrainDataset(KITTISemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path,
            logger=logger
        )
        self.repeat = dataset_cfg.get('PRETRAIN_REPEST', 1)

    def __len__(self):
        return super().__len__() * self.repeat

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch or self.repeat > 1:
            index = index % len(self.kitti_infos)
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict


class KITTILabeledDataset(KITTISemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path,
            logger=logger
        )
        self.labeled_data_for = dataset_cfg.LABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        assert 'annos' in info
        annos = info['annos']
        annos = common_utils.drop_info_with_name(annos, name='DontCare')
        loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
        gt_names = annos['name']
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

        input_dict.update({
            'gt_names': gt_names,
            'gt_boxes': gt_boxes_lidar
        })

        road_plane = self.get_road_plane(sample_idx)
        if road_plane is not None:
            input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.labeled_data_for)
        return tuple([teacher_dict, student_dict])


class KITTIUnlabeledDataset(KITTISemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path,
            logger=logger
        )
        self.unlabeled_data_for = dataset_cfg.UNLABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.unlabeled_data_for)
        return tuple([teacher_dict, student_dict])


class KITTITestDataset(KITTISemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=False, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is False
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path,
            logger=logger
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict