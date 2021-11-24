from .detector3d_template import Detector3DTemplate

class CenterPoints(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_type = 'origin'

    def set_model_type(self, model_type):
        assert model_type in ['origin', 'teacher', 'student']
        self.model_type = model_type
        self.dense_head.model_type = model_type

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.model_type == 'origin':
            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss()

                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
        elif self.model_type == 'teacher':
            return batch_dict
        elif self.model_type == 'student':
            if self.training:
                if 'gt_boxes' in batch_dict:
                    loss, tb_dict, disp_dict = self.get_training_loss()
                    ret_dict = {
                        'loss': loss
                    }
                    return batch_dict, ret_dict, tb_dict, disp_dict
                else:
                    return batch_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
        else:
            raise NotImplementedError('Unsupported model type')

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
