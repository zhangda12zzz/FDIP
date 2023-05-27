for i in range(len(err_on)):
    if tp_err_on[i][0][0] - err_on[i][0][0] > max_sip_dif:
        max_sip_dif = tp_err_on[i][0][0] - err_on[i][0][0]
        max_sip_dif_idx = i
        
a_pre_on = pre_on[i].squeeze(0).numpy() #[t,24,3,3]
a_gt_on = gt_on[i].squeeze(0).numpy()   #[t,24,3,3]
a_tp_pre_on = tp_pre_on[i].squeeze(0).numpy() #[t,24,3,3]
a_tp_gt_on = tp_gt_on[i].squeeze(0).numpy()   #[t,24,3,3]

pre_on_csv_lines = []
gt_on_csv_lines = []
tp_on_csv_lines = []
for f in range(a_pre_on.shape[0]):   #[24,3,3]
    quat = Quaternions.from_transforms(a_pre_on[f]).qs  # [24,4]
    quat = torch.Tensor(quat)
    quat = quat.flatten().numpy()
    pre_on_csv_lines.append(quat)
    
    quat_gt = Quaternions.from_transforms(a_gt_on[f]).qs
    quat_gt = torch.Tensor(quat_gt)
    quat_gt = quat_gt.flatten().numpy()
    gt_on_csv_lines.append(quat_gt)
    
    quat_tp = Quaternions.from_transforms(a_tp_pre_on[f]).qs
    quat_tp = torch.Tensor(quat_tp)
    quat_tp = quat_tp.flatten().numpy()
    tp_on_csv_lines.append(quat_tp)
    
with open("GGIP/eval/forUnity/dip-imu_online_res_mybest.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(pre_on_csv_lines)
with open("GGIP/eval/forUnity/dip-imu_online_gt_mybest.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(gt_on_csv_lines)
with open("GGIP/eval/forUnity/dip-imu_online_tp_mybest.csv","w") as csvfile: 
    writer = csv.writer(csvfile)    
    writer.writerows(tp_on_csv_lines)