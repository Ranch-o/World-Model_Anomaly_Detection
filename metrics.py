import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, confusion_matrix, precision_score, recall_score, precision_recall_curve
# import open3d as o3d

gt_path = "/home/lukasnroessler/Anomaly_Datasets/AnoVox/Scenario_911ef0ea-bec9-4f78-aa3b-a3b83ef07319/VOXEL_GRID/VOXEL_GRID_413.npy"
pred_path = "/home/lukasnroessler/Projects/RbA/voxelpreds/voxelarray_score_0000000011.npy"

# gt_path = "/home/lukasnroessler/Anomaly_Datasets/AnoVox/Scenario_20a82f73-c3e0-47a1-8635-afc6ca3a2d12/VOXEL_GRID/VOXEL_GRID_384.npy"
# gt_path = "/home/lukasnroessler/Anomaly_Datasets/AnoVox/Scenario_388c88a1-c7e1-4937-a8ca-41a31521eab9/VOXEL_GRID/VOXEL_GRID_319.npy"
# pred_path = "/home/lukasnroessler/Projects/RbA/voxelpreds/voxelarray_score_0000000004.npy"


COLOR_PALETTE = (
        np.array(
            [
                (0, 0, 0),          # unlabeled     =   0u
                (128, 64, 128),     # road          =   1u
                (244, 35, 232),     # sidewalk      =   2u
                (70, 70, 70),       # building      =   3u=         bin_preds =
                (102, 102, 156),    # wall          =   4u
                (190, 153, 153),    # fence         =   5u
                (153, 153, 153),    # pole          =   6u
                (250, 170, 30),     # traffic light =   7u
                (220, 220, 0),      # traffic sign  =   8u
                (107, 142, 35),     # vegetation    =   9u
                (152, 251, 152),    # terrain       =  10u
                (70, 130, 180),     # sky           =  11u
                (220, 20, 60),      # pedestrian    =  12u
                (255, 0, 0),        # rider         =  13u
                (0, 0, 142),        # Car           =  14u
                (0, 0, 70),         # truck         =  15u
                (0, 60, 100),       # bus           =  16u
                (0, 80, 100),       # train         =  17u
                (0, 0, 230),        # motorcycle    =  18u
                (119, 11, 32),      # bicycle       =  19u
                (110, 190, 160),    # static        =  20u
                (170, 120, 50),     # dynamic       =  21u
                (55, 90, 80),       # other         =  22u
                (45, 60, 150),      # water         =  23u
                (157, 234, 50),     # road line     =  24u
                (81, 0, 81),        # ground         = 25u
                (150, 100, 100),    # bridge        =  26u
                (230, 150, 140),    # rail track    =  27u
                (180, 165, 180),    # guard rail    =  28u
                (250, 128, 114),    # home          =  29u
                (255, 36, 0),       # animal        =  30u
                (224, 17, 95),      # nature        =  31u
                (184, 15, 10),      # special       =  32u
                (245, 0, 0),        # airplane      =  33u
                (245, 0, 0),        # falling       =  34u
            ]
        )
)  # normalize each channel [0-1] since this is what Open3D uses


def mask_intersect(preds, gts):
    # preds_, gts_ = voxelgrid_wrong_padding(preds, gts) # outputs here are joined
    preds_, gts_ = voxelgrid_intersect(preds, gts)

    preds_scores, gts_scores = preds_[:,-1:], gts_[:,-1:]
    preds_scores, gts_scores = np.squeeze(preds_scores), np.squeeze(gts_scores)

    # preds_ = preds_.squeeze()
    # gts_ = gts_.squeeze()

    # full_gts_grid = np.concatenate([joined_grids, gts])
    # full_gts_grid = np.unique(full_gts_grid, axis = 0)=         bin_preds =

    ood_mask, ind_mask = mask_anomaly_voxels(gts_scores)


    ood_out = preds_scores[ood_mask]
    ind_out = preds_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    return val_out, val_label


def calculate_aupr(preds, gts):

    preds_, gts_ = voxelgrid_padding(preds, gts) # outputs here are joined
    # preds_, gts_ = voxelgrid_intersect(preds, gts)

    # preds_ = preds_.squeeze()
    # gts_ = gts_.squeeze()

    # full_gts_grid = np.concatenate([joined_grids, gts])
    # full_gts_grid = np.unique(full_gts_grid, axis = 0)

    ood_mask, ind_mask = mask_anomaly_voxels(gts_)


    ood_out = preds_[ood_mask]
    ind_out = preds_[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))


    return average_precision_score(val_label, val_out)

def calculate_auroc(preds, gts):

    fpr, tpr, threshold = roc_curve(gts, preds)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0
    # print('Started FPR search.')
    for i, j, k in zip(tpr, fpr, threshold):
        if i > 0.95:
            fpr_best = j
            break
    # print(k)
    return roc_auc, fpr_best, k




def calculate_specificity(preds, gts, thresholds: list=[0.5]):
    specificity_scores = []
    for i, threshold in enumerate(thresholds):
        preds_bin = np.where(preds > threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(gts, preds_bin).ravel()
        specificity = tn / (tn+fp)
        specificity_scores.append(specificity)
    return np.mean(specificity_scores)

def calculate_f1(preds, gts, thresholds: list=[0.5]):
    f1_scores = []
    for i, threshold in enumerate(thresholds):
        preds_bin = np.where(preds > threshold, 1, 0)
        f1 = f1_score(gts, preds_bin)
        f1_scores.append(f1)
    # tn, fp, fn, tp = confusion_matrix(gts, preds).ravel()
    # f1 = (2 * tp) / (2 * tp + fn + fp)
    return np.mean(f1_scores)


def calculate_threshold(preds, gts, delta_thresholds: list=[0.5, 0.7, 0.9]): # Im not sure if taking average is ok
    """
    approach from segmentmeifyoucan to calculate the optimal threshold
    delta* = argmax 2 * precision(delta) * recall(delta) / (precision(delta) + recall(delta))
    """
    new_deltas = []
    for _, threshold in enumerate(delta_thresholds):
        bin_preds = np.where(preds > threshold, 1, 0)

        # precision = precision_score(gts, bin_preds)
        # recall = recall_score(gts, bin_preds)
        # delta_ = 2 * precision * recall / (precision + recall)
        # new_deltas.append(delta_)
    return np.mean(new_deltas)


def calculate_threshold_from_prc(preds, gts):
    precision, recall, thresholds = precision_recall_curve(gts, preds)
    # minimize distance between precision and recall scores
    optimal_threshold = sorted(list(zip(np.abs(precision - recall), thresholds)), key=lambda i: i[0], reverse=False)[0][1]
    return optimal_threshold


def mask_anomaly_voxels(gts):
    gt_array = np.copy(gts)
    # anomaly_mask = (gts == 33)# or gts == 34)
    # inlier_mask = (gts != 33 )#and gts != 34)
    anomaly_mask = (gts == 29)
    inlier_mask = (gts != 29)

    # anomaly_mask = (gts == 20) # nur für das example jetzt
    # inlier_mask = (gts != 20)

    return anomaly_mask, inlier_mask


def voxelgrid_merge(preds, gts):
    predscores = preds[:, -1:]

    preds_size, _ = preds.shape
    gts_size, _ = gts.shape

    pred_voxels = preds[:, :3].astype(np.uint16)

    gt_voxels = gts[:,:3].astype(np.uint16)

    gt_idx = gts[:,-1:].astype(np.uint16)

    concat_voxels = np.concatenate((pred_voxels, gt_voxels))
    print(concat_voxels.shape)
    joined_voxel_grid = np.unique(concat_voxels, axis=0)
    print("joined voxels shape", joined_voxel_grid.shape)
    # joined_mask = np.all(joined_voxel_grid==pred_voxels,axis=1)
    joined_voxel_grid_preds = np.c_[joined_voxel_grid, np.zeros((joined_voxel_grid.shape[0],1))]
    joined_voxel_grid_gts = np.copy(joined_voxel_grid_preds)
    for i, coordinate in enumerate(joined_voxel_grid):

        # pred_voxel_index = np.where((pred_voxels == coordinate).all(axis=1))[0]
        # if pred_voxel_index.size != 0: # if this voxel is also in the predictions grid
        #     anomaly_score = predscores[pred_voxel_index][0]
        #     joined_voxel_grid_preds[i][3] = anomaly_score # set the voxel value to the prediction score

        gts_voxel_index = np.where((gt_voxels == coordinate).all(axis=1))[0]
        if gts_voxel_index.size != 0:
            gt_id = gt_idx[gts_voxel_index][0]
            joined_voxel_grid_gts[i][3] = gt_id
    # joined_voxel_grid = np.c_[joined_voxel_grid, np.zeros((joined_voxel_grid.shape[0],1))]

    # joined_voxel_grid[preds_mask][3] = predscores[]
    print("joined voxel grid shape after masking", joined_voxel_grid.shape)
    # joined_voxel_grid_values = joined_voxel_grid_preds[:,-1:]
    # unique_elems = np.unique(joined_voxel_grid_values)
    # np.save("joinedpreds.npy", joined_voxel_grid_preds)
    np.save("debugjoin.npy", joined_voxel_grid_gts)
    return joined_voxel_grid_preds, joined_voxel_grid_gts


def voxelgrid_padding(preds, gts):
    preds_size, _ = preds.shape
    gts_size, _ = gts.shape

    pred_scores = preds[:, -1:].astype(np.float64)
    pred_voxels = preds[:, :3].astype(np.uint16)
    gt_voxels = gts[:,:3].astype(np.uint16)
    gt_labels = gts[:,-1:].astype(np.uint16)

    intersect_indices = np.where((gt_voxels==pred_voxels[:,None]).all(-1))[1]
    intersected_grid = gt_voxels[intersect_indices]

    concat_grid = np.concatenate([intersected_grid, pred_voxels])
    leftjoin_grid = np.unique(concat_grid, axis=0)
    leftjoin_labels = np.c_[leftjoin_grid, np.zeros((leftjoin_grid.shape[0],1))]
    leftjoin_preds = np.copy(leftjoin_labels)
    for i, coordinate in enumerate(leftjoin_grid):
        pred_voxel_index = np.where((np.isclose(pred_voxels, coordinate)).all(axis=1))[0]
        if pred_voxel_index.size != 0: # if this voxel is also in the predictions grid
            anomaly_score = pred_scores[pred_voxel_index][0]
            leftjoin_preds[i][3] = anomaly_score # set the voxel value to the prediction score

        gts_voxel_index = np.where((np.isclose(gt_voxels, coordinate)).all(axis=1))[0]
        if gts_voxel_index.size != 0:
            gt_id = gt_labels[gts_voxel_index][0]
            leftjoin_labels[i][3] = gt_id

    np.save("leftjoinpreds.npy", leftjoin_preds)
    np.save("leftjoingts.npy", leftjoin_labels)

    return leftjoin_preds, leftjoin_labels


def voxelgrid_intersect(preds, gts): # only grid of prediction is relevant for evaluation
    gt_labels = gts[:,-1:].astype(np.float64) # to match pred voxels and values
    gt_voxels = gts[:,:3].astype(np.float64)

    pred_scores = preds[:,-1:]
    pred_voxels = preds[:,:3] #.astype(np.uint16)

    size, dim = gt_voxels.shape

    # dtype={'names':['f{}'.format(i) for i in range(dim)],
    #    'formats':dim * [gt_voxels.dtype]}

    # intersect_grid = np.intersect1d(gt_voxels.view(dtype), pred_voxels.view(dtype))

    # # This last bit is optional if you're okay with "intersect_grid" being a structured array...
    # intersect_grid = intersect_grid.view(gt_voxels.dtype).reshape(-1, dim)

    # intersect_grid = np.where((pred_voxels==gt_voxels).all(axis=1))
    intersect_indices = np.where((gt_voxels==pred_voxels[:,None]).all(-1))[1]

    intersect_grid = gt_voxels[intersect_indices]
    intersect_labels = np.c_[intersect_grid, np.zeros((intersect_grid.shape[0],1))]
    intersect_preds = np.copy(intersect_labels)

    for i, coordinate in enumerate(intersect_grid):
        pred_voxel_index = np.where((np.isclose(pred_voxels, coordinate)).all(axis=1))[0]
        if pred_voxel_index.size != 0: # if this voxel is also in the predictions grid
            anomaly_score = pred_scores[pred_voxel_index][0]
            intersect_preds[i][3] = anomaly_score # set the voxel value to the prediction score

        gts_voxel_index = np.where((np.isclose(gt_voxels, coordinate)).all(axis=1))[0]
        if gts_voxel_index.size != 0:
            gt_id = gt_labels[gts_voxel_index][0]
            intersect_labels[i][3] = gt_id

    # np.save("intersectpreds.npy", intersect_preds)
    # np.save("intersectgts.npy", intersect_labels)

    # voxel_pcd = o3d.geometry.PointCloud()
    # voxel_pcd.points = o3d.utility.Vector3dVector(intersect_labels[:,:3])
    # debug_colors = intersect_labels[:,-1:].squeeze().astype(np.uint8)
    # voxel_colors = np.squeeze(COLOR_PALETTE[debug_colors]) / 255.0

    # voxel_pcd.colors = o3d.utility.Vector3dVector(voxel_colors)
    # voxel_world = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_pcd, 0.2)
    # o3d.visualization.draw_geometries([voxel_world])

    return intersect_preds, intersect_labels


def anomaly_included(gts):
    # gt_labels = gts[:, -1]
    # gt_labels = gt_labels.squeeze()
    # for label in gt_labels:
    #     if label == 33 or label == 34:
    #         return True
    # return False
    return np.isin(gts, 1).any()

def intersect_grids(pred_path, gt_path, front_only:bool=False, return_anomaly_detectable:bool=True):# , iteration):
    gt = np.load(gt_path)
    pred = np.load(pred_path)

    if front_only:
        # x < 500 is behind ego vehicle
        front_voxels = np.where(gt[:,0] > 500)
        gt = gt[front_voxels]

    pred_bin, gt_bin = mask_intersect(pred, gt)

    # np.save("pred_bin{}.npy".format(str(iteration)), pred_bin)
    # np.save("gt_bin{}.npy".format(str(iteration)), gt_bin)

    # pred_bin = pred_bin # + 1

    # print(pred_bin.shape)
    # print(gt_bin.shape)

    # ap = calculate_aupr(pred_bin, gt_bin)
    # ap = average_precision_score(gt_bin, pred_bin)
    # auroc, fpr, _ = calculate_auroc(pred_bin, gt_bin)
    # result = {
    #     'auroc': auroc,
    #     'aupr': ap,
    #     'fpr95': fpr
    # }

    return (pred_bin, gt_bin, anomaly_included(gt_bin)) if return_anomaly_detectable else (pred_bin, gt_bin)



def compute_metrics(preds, gts, ignore=[]):
    ap = average_precision_score(gts, preds)
    auroc, fpr, _ = calculate_auroc(preds, gts)
    # f1 = f1_score(gts, preds)
    # threshold = calculate_threshold(preds, gts, delta_thresholds=[fpr]) # fishyscapes way to calculate threshold
    # f1_threshold = 1 - fpr # stupid number but: based on that f1 paper, f1 threshold is approx 1/2 of the f1 score that it achieves
    # preds_f1 = np.where(preds > f1_threshold, 1, 0)
    # f1 = f1_score(gts, preds_f1) # calculate_f1(preds_discrete, gts)
    smiyc_thresholds = np.arange(start=0.25, stop=0.75, step=0.05)
    prc_threshold = calculate_threshold_from_prc(preds, gts)
    # print("prc threshold: ", prc_threshold)
    # thresholds = [0.5, 0.7, 0.9]

    f1 = calculate_f1(preds, gts, thresholds=[prc_threshold])
    # preds_fpr = np.where(preds > fpr, 1, 0)
    specificity_scores = []
    precision_scores = []
    for i, threshold in enumerate(smiyc_thresholds):
        preds_bin = np.where(preds > threshold, 1, 0)
        cm_values = confusion_matrix(gts, preds_bin).ravel()
        tn, fp, fn, tp = cm_values
        # tn, fp, fn, tp = confusion_matrix(gts, preds_bin).ravel()
        specificity_i = tn / (tn+fp)
        specificity_scores.append(specificity_i)
        precision_i = tp / (tp + fp)
        precision_scores.append(precision_i)
    specificity = np.mean(specificity_scores)
    precision_score = np.mean(precision_scores)

    # specificity = calculate_specificity(preds, gts, [prc_threshold])
    result = {
        'auroc': float(auroc),
        'aupr': float(ap),
        'fpr95': float(fpr),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'ppv': float(precision_score),
    }
    return result


if __name__ == "__main__":
    gt = np.load(gt_path)
    pred = np.load(pred_path)

    pred_bin, gt_bin = mask_intersect(pred, gt)

    # print(pred_bin.shape)
    print("shape of intersection:", gt_bin.shape)

    # ap = calculate_aupr(pred_bin, gt_bin)
    # ap = average_precision_score(gt_bin, pred_bin)
    # auroc, fpr, _ = calculate_auroc(pred_bin, gt_bin)
    # result = {
    #     'auroc': auroc,
    #     'aupr': ap,
    #     'fpr95': fpr
    # }
    # print(ap)
    # print(auroc)
    # print(fpr)
    result = compute_metrics(pred_bin, gt_bin)
    for key in result.keys():
        print(key, ": ", result[key])
    # for input in inputs:
    #     print(f"## Opening {input} ##")
    #     if check_file_ending(input):
    #         transform_npy(input)
    #     else:
    #         print("     --> wrong filetype")
