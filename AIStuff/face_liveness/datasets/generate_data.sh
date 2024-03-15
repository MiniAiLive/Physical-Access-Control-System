rm -rf ./motion_analysis/annotation
mkdir ./motion_analysis/annotation
rm -rf ./motion_analysis/image_data
mkdir ./motion_analysis/image_data

python3 video_jpg_ucf101_hmdb51.py
python3 n_frames_ucf101_hmdb51.py
python3 gen_anns_list.py
python3 ucf101_json.py
