import performance

if __name__ == '__main__':
    # test model segmentation performance
    pred_path = r''  # predict result path
    gt_path = r''  # ground truth path
    calc_performance(pred_path, gt_path, img_resize=(1400, 1400))
