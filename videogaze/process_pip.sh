source activate videopipelineenv
echo $1','$2 > configuration.csv
bash cleanall.sh
python shot.py
python detect.py --method=tiny_faces --tinyFaces_scale=360 --detection_interval=0.25
python tracking.py
python crop.py
python extract.py
python main.py
