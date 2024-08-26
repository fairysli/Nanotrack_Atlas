import os
import cv2
import time
import numpy as np
import acl
import argparse
from glob import glob
from models.atlas_tracker import NanoTracker_Atlas

from core.config import cfg

parser = argparse.ArgumentParser(description='tracking demo')

parser.add_argument('--config', default='./models/configv3.yaml', type=str, help='config file')

parser.add_argument('--save', default=True, action='store_true', help='whether visualzie result')

parser.add_argument('--video_name', default='./video/girl_dance.mp4', type=str, help='videos or image files')

args = parser.parse_args()

DEVICE_ID = 0  

def init_acl(device_id):
    acl.init()
    ret = acl.rt.set_device(device_id)
    if ret:
        raise RuntimeError(ret)
    context, ret = acl.rt.create_context(device_id)
    if ret:
        raise RuntimeError(ret)
    print('Init ACL Successfully')
    return context


def deinit_acl(context, device_id):
    ret = acl.rt.destroy_context(context)
    if ret: 
        raise RuntimeError(ret)
    ret = acl.rt.reset_device(device_id)
    if ret: 
        raise RuntimeError(ret)
    ret = acl.finalize() 
    if ret:
        raise RuntimeError(ret)
    print('Deinit ACL Successfully')

def get_frames(video_name): 
    if not video_name:
        cap = cv2.VideoCapture(0) 
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break 

    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name)
        
        for i in range(50):
            cap.read()

        while True:
            ret, frame = cap.read()
            if ret:
                yield frame 
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


 
def main():
    cfg.merge_from_file(args.config)
    
    Tback_weight = './weights/nanotrack_T1.om'
    Xback_weight = './weights/nanotrack_X1.om'
    Head_weight = './weights/nanotrack_H1.om'

    tracker = NanoTracker_Atlas(Tback_weight, Xback_weight, Head_weight)
    first_frame = True

    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    for frame in get_frames(args.video_name):
        if first_frame:
            if args.save: 
                if args.video_name.endswith('avi') or \
                        args.video_name.endswith('mp4') or \
                        args.video_name.endswith('mov'):
                        cap = cv2.VideoCapture(args.video_name)
                        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                else:
                    fps = 30 
            
                save_video_path = args.video_name.split(video_name)[0] + video_name + '_tracking.mp4'
                print(save_video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (frame.shape[1], frame.shape[0])
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                init_rect = [275, 149, 62, 60] # girl_dance.mp4
                print(init_rect)

            except:
                print(f"Exception in ROI selection: {e}")
                exit()
            
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            t1 = time.time()
            outputs = tracker.track(frame)
            print(outputs)
            # print('fps：', 1. / (time.time() - t1))
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)
            
            # cv2.imshow(video_name, frame)
            # cv2.waitKey(30)
            # cv2.imwrite(os.path.join(img_savedir, '%03d.jpg'%count), frame)
            # count += 1
        if args.save:
            # print("video_writer.write \n")
            video_writer.write(frame)
    
    if args.save:
        # print("video_writer.release \n")
        video_writer.release()
    tracker.release()

if __name__ == '__main__':
    context = init_acl(DEVICE_ID)
    main()
    deinit_acl(context, 0)