import acl
from utils.bbox import corner2center
from core.config import cfg

NPY_FLOAT32 = 11
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_DEVICE, ACL_HOST = 0, 1
ACL_SUCCESS = 0
ACL_MEM_MALLOC_NORMAL_ONLY = 2

import numpy as np
import cv2


class NanoTracker_Atlas(object):
    def __init__(self, Tback_weight, Xback_weight, Head_weight):
        self.Tback_weight = Tback_weight
        self.Xback_weight = Xback_weight
        self.Head_weight = Head_weight
        # self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.score_size =cfg.TRACK.OUTPUT_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self._is_released = False
        self.init_resources()

    def init_resources(self):
        self.tback_model, self.tback_model_desc = self.load_model(self.Tback_weight)
        self.xback_model, self.xback_model_desc = self.load_model(self.Xback_weight)
        self.head_model, self.head_model_desc = self.load_model(self.Head_weight)
        
        print('Model init resource stage success')


    def load_model(self, model_path):
        model_id, ret = acl.mdl.load_from_file(model_path)
        model_desc = acl.mdl.create_desc()
        ret  = acl.mdl.get_desc(model_desc, model_id)
        # if ret != ACL_SUCCESS:
        #     raise Exception(f"Load model from {model_path} failed, error code: {ret}")
        print("Model init resource stage success")
        return model_id, model_desc   
    
    def release(self):
        if self._is_released:
            return

        print("Model start release...")
        if self.tback_model:
            ret = acl.mdl.unload(self.tback_model)
        if self.xback_model:
            ret = acl.mdl.unload(self.xback_model)
        if self.head_model:
            ret = acl.mdl.unload(self.head_model)
        if self.tback_model_desc:
            ret = acl.mdl.destroy_desc(self.tback_model_desc)
        if self.xback_model_desc:
            ret = acl.mdl.destroy_desc(self.xback_model_desc)
        if self.head_model_desc:
            ret = acl.mdl.destroy_desc(self.head_model_desc)
        self._is_released = True
        print("Model release source success")

    def _release_dataset(self, dataset):
        ''' 释放 aclmdlDataset 类型数据 '''
        if not dataset:
            return
        num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf:
                ret = acl.destroy_data_buffer(data_buf)
        ret = acl.mdl.destroy_dataset(dataset)

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points
    
    def _convert_bbox(self, delta, point):
        delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)
        delta = np.copy(delta)
        delta[0, :] = point[:, 0] - delta[0, :] #x1
        delta[1, :] = point[:, 1] - delta[1, :] #y1
        delta[2, :] = point[:, 0] + delta[2, :] #x2
        delta[3, :] = point[:, 1] + delta[3, :] #y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.transpose(1, 2, 3, 0).reshape(4, -1)
            score = 1 / (1 + np.exp(-score))
        else:
            score = score.transpose(1, 2, 3, 0).reshape(self.cls_out_channels, -1).transpose(1, 0)
            score = self._softmax(score, axis=1)[:, 1]
        return score
    
    def _softmax(self, x, axis=None):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


    def _convert_bbox_numpy(self, delta, point):
        delta = delta.transpose((1,2,3,0)).reshape(4, -1)
        result = np.zeros_like(delta)
        result[0, :] = point[:, 0] - delta[0, :]  # x1
        result[1, :] = point[:, 1] - delta[1, :]  # y1
        result[2, :] = point[:, 0] + delta[2, :]  # x2
        result[3, :] = point[:, 1] + delta[3, :]  # y2
        result[0, :], result[1, :], result[2, :], result[3, :] = corner2center(delta)
        return result

    def _convert_score_numpy(self, score):
        def sofmax(logits):
            e_x = np.exp(logits)
            probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
            return probs

        score = score.transpose((1,2,3,0)).reshape(self.cls_out_channels, -1).transpose((1,0))
        score = sofmax(score)[:,1]

        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)

        return im_patch

    def init(self, img, bbox):
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
        back_T_in = z_crop
        self.Toutput = self.run_model(self.tback_model, self.tback_model_desc, [back_T_in])


    def track(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE, round(s_x), self.channel_average)

        back_X_in = x_crop
        self.Xoutput = self.run_model(self.xback_model, self.xback_model_desc, [back_X_in])

        head_T_in = self.Toutput[0]
        head_X_in = self.Xoutput[0]
        outputs = self.run_model(self.head_model, self.head_model_desc, [head_T_in, head_X_in])
        score = self._convert_score(outputs[0])
        pred_bbox = self._convert_bbox(outputs[1], self.points)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)

        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]
        return {
            'bbox': bbox,
            'best_score': best_score
        }
    

    def run_model(self, model_id, model_desc, input_list):
        input_num = acl.mdl.get_num_inputs(model_desc)
        input_dataset = acl.mdl.create_dataset()
        for i in range(input_num):
            item = input_list[i]
            data_ptr = acl.util.bytes_to_ptr(item.tobytes())
            size = item.size * item.itemsize
            dataset_buffer = acl.create_data_buffer(data_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(input_dataset, dataset_buffer)
        if ret != ACL_SUCCESS:
            self._release_dataset(input_dataset)
        # print("Create model input dataset success")
        
        output_size = acl.mdl.get_num_outputs(model_desc)
        output_dataset = acl.mdl.create_dataset()
        for i in range(output_size):
            size = acl.mdl.get_output_size_by_index(model_desc, i)
            buf, ret = acl.rt.malloc(size,  ACL_MEM_MALLOC_NORMAL_ONLY)
            dataset_buffer = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(output_dataset, dataset_buffer)
        if ret != ACL_SUCCESS:
            # acl.rt.free(buf)
            # acl.destroy_data_buffer(dataset_buffer)
            self._release_dataset(output_dataset)
        # print("Create model output dataset success")

        ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
        model_output = self._output_dataset_to_numpy(model_desc, output_dataset, output_size)
        
        self._release_dataset(input_dataset)
        input_dataset = None
        self._release_dataset(output_dataset)
        output_dataset = None
        
        return model_output
    
    def _unpack_bytes_array(self, byte_array, shape, datatype):
        np_type = None

        if datatype == 0:  # ACL_FLOAT
            np_type = np.float32
        elif datatype == 1:  # ACL_FLOAT16
            np_type = np.float16
        elif datatype == 3:  # ACL_INT32
            np_type = np.int32
        elif datatype == 8:  # ACL_UINT32
            np_type = np.uint32
        else:
            print("unsurpport datatype ", datatype)
            return

        return np.frombuffer(byte_array, dtype=np_type).reshape(shape)
    
    def _output_dataset_to_numpy(self, model_desc, output_dataset, _output_num):
        dataset = []
        for i in range(_output_num):
            buffer = acl.mdl.get_dataset_buffer(output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buffer)
            size = acl.get_data_buffer_size(buffer)
            narray = acl.util.ptr_to_bytes(data_ptr, size)

            dims = acl.mdl.get_output_dims(model_desc, i)[0]["dims"]
            datatype = acl.mdl.get_output_data_type(model_desc, i)
            output_nparray = self._unpack_bytes_array(narray, tuple(dims), datatype)
            dataset.append(output_nparray)
        return dataset
