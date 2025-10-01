train_on_dota.py  to train initial model

train_transfer.py  use above model to train further on the cranes dataset

eval_transfer.py to test model on certain images. code options for both txt and visual result

eval_transfer_group.py compute all k=0,1,2,3 rotations and take average of overlapping boxes

compare_gt_and_pred_boxes.py  coda to compare pred and gt txts and find FPs FNs TPs
