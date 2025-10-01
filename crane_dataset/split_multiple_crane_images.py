##ta images me ta multiple cranes ta kanw split opws kai to dota

from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="./CRANE_dataset/",
    save_dir="./CRANE_dataset-split/",
    rates=[1.0],  # multiscale
    gap=200,
    crop_size=640
)
