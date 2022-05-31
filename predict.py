from DLAnet import DlaNet
import torch
import numpy as np
from utils import *
from dataset_test import ctDataset
import os
from torch.utils.data import DataLoader
import pdb
import cv2
import argparse

run_name = "zjn-cennet"


class Predictor:
    def __init__(self, use_gpu):
        # Todo: the mean and std need to be modified according to our dataset
        # self.mean_ = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746], \
        #                 dtype=np.float32).reshape(1, 1, 3)
        # self.std_  = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], \
        #                 dtype=np.float32).reshape(1, 1, 3)

        # input image size
        self.inp_width_ = 512
        self.inp_height_ = 512
        self.number_detection = 100
        # confidence threshold
        self.thresh_ = 0.35

        self.use_gpu_ = use_gpu

    def nms(self, heat, kernel=3):
        ''' Non-maximal supression
        '''
        pad = (kernel - 1) // 2
        hmax = torch.nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        # hmax == heat when this point is local maximal
        keep = (hmax == heat).float()
        return heat * keep

    def find_top_k(self, heat, K):
        ''' Find top K key points (centers) in the headmap
        '''

        batch, cat, height, width = heat.size()
        # pdb.set_trace()
        heat_nms = torch.where(heat > 0.12, heat, torch.full_like(heat, 0))  # 0.09
        last_index = heat_nms.sum(dim=3)  # [4, 256]

        heat_centerline = torch.sum(heat_nms, dim=3)  # [batch,cat,height]
        # heat_centerline = heat_centerline / (last_index + 1e-5)
        # heat_centerline = torch.mean(heat,dim=3) #[batch,cat,height]
        topk_scores, topk_inds_height = torch.topk(heat_centerline, K)  # topk_score torch.Size([1, 6, 100])
        topk_inds = topk_inds_height % (height)  # topk_inds.shape  torch.Size([1, 6, 100])
        topk_ys = topk_inds  # topk_ys.shape  torch.Size([1, 6, 100])
        # topk_xs = (topk_inds % width).int().float()
        # pdb.set_trace()
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)  # topk_ind.shape torch.Size([1, 100])
        topk_clses = (topk_ind / K)  # torch.Size([1, 100])
        # topk_inds = gather_feat_(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_height_cat_one = gather_feat_(topk_inds_height.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys_final = gather_feat_(topk_ys.int().float().view(batch, -1, 1), topk_ind).view(batch,
                                                                                              K)  # topk_ys.shape [1, 100]
        # for center_x
        topk_height_cat_one_channel = topk_height_cat_one + topk_clses * height  # [1,100]
        topk_k_width = gather_feat_clses(heat_nms, topk_height_cat_one_channel)  # torch.Size([1, 100, 256])
        # last_index = torch.where(topk_k_width > 0,torch.full_like(topk_k_width,1),torch.full_like(topk_k_width, 0))
        # pdb.set_trace()
        _, topk_2th_max = torch.topk(topk_k_width, k=2, dim=2)  # [1,100,2]
        # last_index = last_index.sum(dim=2).squeeze(dim=0)
        # last_item = [int(topk_k_width[index][int(x.item()) - 1].item()) for index, x in enumerate(last_index)]
        # index_all = torch.nonzero(topk_k_width,as_tuple=False)
        topk_xs_s = torch.ones_like(topk_ys_final)
        topk_xs_e = torch.ones_like(topk_ys_final)
        inds_w = torch.ones([batch, K, 2]).to(device)
        for j in range(batch):
            for index in range(len(topk_2th_max.squeeze(dim=0))):
                # pdb.set_trace()
                if topk_k_width[j, index, topk_2th_max[j, index, 0]] > topk_k_width[
                    j, index, topk_2th_max[j, index, 1]]:
                    topk_xs_s[j, index] = topk_2th_max[j, index, 1]
                    topk_xs_e[j, index] = topk_2th_max[j, index, 0]
                else:
                    topk_xs_s[j, index] = topk_2th_max[j, index, 0]
                    topk_xs_e[j, index] = topk_2th_max[j, index, 1]
        # pdb.set_trace()
        inds_w[:, :, 0] = topk_ys_final * width + topk_xs_s
        inds_w[:, :, 1] = topk_ys_final * width + topk_xs_e
        return topk_score, inds_w.long(), topk_clses.int(), topk_ys_final, topk_xs_s, topk_xs_e

    def pre_process(self, image):
        ''' Preprocess the image

            Args:
                image - the image that need to be preprocessed
            Return:
                images (tensor) - images have the shape (1，3，h，w)
        '''
        height = image.shape[0]
        width = image.shape[1]

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        # shrink the image size and normalize here
        inp_image = cv2.resize(image, (self.inp_width_, self.inp_height_))

        plt.imshow(cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB))
        plt.show()

        # inp_image = ((inp_image / 255. - self.mean_) / self.std_).astype(np.float32)
        inp_image = (inp_image / 255.).astype(np.float32)

        # from three to four dimension
        # (h, w, 3) -> (3, h, w) -> (1，3，h，w)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, self.inp_height_, self.inp_width_)
        images = torch.from_numpy(images)

        return images

    def post_process(self, xs, ys, wh, reg):
        ''' (Will modify args) Transfer all xs, ys, wh from heatmap size to input size
        '''
        for i in range(xs.size()[1]):
            xs[0, i, 0] = xs[0, i, 0] * 2
            ys[0, i, 0] = ys[0, i, 0] * 2
            wh[0, i, 0] = wh[0, i, 0] * 2
            wh[0, i, 1] = wh[0, i, 1] * 2

    def ctdet_decode(self, heads, original_size, K=40):
        ''' Decoding the output

            Args:
                heads ([heatmap, width/height, regression]) - network results
            Return:
                detections([batch_size, K, [xmin, ymin, xmax, ymax, score]])
        '''
        heat, h_ud, reg_y = heads
        down_ratio = original_size / heat.shape[3]
        batch, cat, height, width = heat.size()

        if (not self.use_gpu_):
            plot_heapmap(heat[0, 0, :, :])

        heat = self.nms(heat)

        if (not self.use_gpu_):
            plot_heapmap(heat[0, 0, :, :])
        #pdb.set_trace()
        scores, inds, clses, ys, x_s, x_e = self.find_top_k(heat, K)
        reg_y = transpose_and_gather_feat(reg_y, inds)
        reg_y = reg_y.view(batch, K, 1)
        ys = ys.view(batch, K, 1) + reg_y

        h_ud = transpose_and_gather_feat(h_ud, inds)
        h_ud = h_ud.view(batch, K, 1)
        x_s, x_e = x_s.view(batch, K, 1), x_e.view(batch, K, 1)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        #pdb.set_trace()
        bboxes = torch.cat([x_s, ys,x_e,ys + h_ud], dim=2)
        #bbox[:, 1] = ys + h_d
        #bbox[:, 3] = ys + h_u
        #bbox[:, 0] = x_s
        #bbox[:, 2] = x_e
        bboxes *= down_ratio
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections

    def draw_bbox(self, image, detections):
        ''' Given the original image and detections results (after threshold)
            Draw bounding boxes on the image
        '''
        height = image.shape[0]
        width = image.shape[1]
        inp_image = cv2.resize(image, (self.inp_width_, self.inp_height_))
        for i in range(detections.shape[0]):
            cv2.rectangle(inp_image, \
                          (detections[i, 0], detections[i, 1]), \
                          (detections[i, 2], detections[i, 3]), \
                          (0, 255, 0), 1)

        original_image = cv2.resize(inp_image, (width, height))

        return original_image

    def process(self, images):
        ''' The prediction process

            Args:
                images - input images (preprocessed)
            Returns:
                output - result from the network
        '''
        with torch.no_grad():
            output = model(images)
            original_size = images.shape[3]
            hm = output['hm'].sigmoid_()
            wh = output['h_ud']
            reg = output['reg_y']

            # Generate GT data for testing
            # hm, wh, reg = generate_gt_data(400)

            heads = [hm, wh, reg]
            if (self.use_gpu_):
                torch.cuda.synchronize()
            dets = self.ctdet_decode(heads, original_size, self.number_detection)  # K is the number of remaining instances

        return output, dets

    def input2image(self, detection, image):
        ''' Transform the detections results from input coordinate (512*512) to original image coordinate

            x is in width direction, y is height
        '''
        default_resolution = [image.shape[1], image.shape[2]]
        det_original = np.copy(detection)
        x0 = det_original[:, 0] / self.inp_width_ * default_resolution[1]
        x0[x0 < 0] = 0
        det_original[:, 0] = x0.astype(np.int16)
        det_original[:, 2] = (det_original[:, 2] / self.inp_width_ * default_resolution[1]).astype(np.int16)
        det_original[:, 1] = (det_original[:, 1] / self.inp_height_ * default_resolution[0]).astype(np.int16)
        det_original[:, 3] = (det_original[:, 3] / self.inp_height_ * default_resolution[0]).astype(np.int16)

        return det_original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    args = parser.parse_args()
    logger.info('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join(args.logging_dir, run_name + '-model_best.pth'))
    use_gpu = torch.cuda.is_available()
    print("Use CUDA? ", use_gpu)

    model = DlaNet(34)
    device = None
    if (use_gpu):
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        # model = nn.DataParallel(model)
        # print('Using ', torch.cuda.device_count(), "CUDAs")
        print('cuda', torch.cuda.current_device(), torch.cuda.device_count())
        device = torch.device('cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
    else:
        device = torch.device('cpu')
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # get the input from the same data loader
    test_dataset = ctDataset(split="test")
    test_dataset_len = test_dataset.__len__()
    print("test_dataset_has ", test_dataset_len, " images.")
    torch.manual_seed(42)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    my_predictor = Predictor(use_gpu)

    for i, sample in enumerate(test_loader):
        if use_gpu:
            for k in sample:
                if k != 'ori_index':
                   sample[k] = sample[k].to(device=device, non_blocking=True)

        # predict the output
        # use the dataloader result instead of do the preprocess again
        output, dets = my_predictor.process(sample['input'])

        # transfer to numpy, and reshape [batch_size, K, 5] -> [K, 5]
        # only considered batch size 1 here
        dets_np = dets.detach().cpu().numpy()[0]

        # select detections above threshold
        threshold_mask = (dets_np[:, -2] > my_predictor.thresh_)
        dets_np = dets_np[threshold_mask, :]

        # need to convert from heatmap coordinate to image coordinate

        # write results to list of txt files
        dets_original = my_predictor.input2image(dets_np, sample['image'])
        # print("Result: ", dets_original)

        # draw the result
        original_image = sample['image'][0].cpu().numpy()
        # pdb.set_trace()
        for i in range(dets_original.shape[0]):
            cv2.rectangle(original_image, \
                          (int(dets_original[i, 0]), int(dets_original[i, 1])), \
                          (int(dets_original[i, 2]), int(dets_original[i, 3])), \
                          (0, 255, 0), 1)
        # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        # plt.show()

        # write the result
        file_path = './results/'
        index_str = sample['ori_index']
        index_str = str(index_str).replace("['", "")
        index_str = index_str.replace("']", "")
        # index_str = '0' * (6 - len(index_str)) + index_str
        cv2.imwrite("./predicts_valid/" + index_str + ".jpg", original_image)

        f = open(file_path + index_str + '.txt', "w")
        f.close()
        # pdb.set_trace()
        dets_original = dets_original[np.lexsort((dets_original[:, 1],))]
        for line in range(dets_original.shape[0]):
            f = open(file_path + index_str + '.txt', "a")
            f.write(str(int(dets_original[line, 5]) + 1) + " " + \
                    str(dets_original[line, 1]) + " " + \
                    str(dets_original[line, 0]) + " " + \
                    str(dets_original[line, 3]) + " " + \
                    str(dets_original[line, 2]) + " " + \
                    str(dets_original[line, 4]) + '\n')

            f.close()

