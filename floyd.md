convert bayeux.jpg -resize x256 -crop 256x256 parts.jpg

convert bayeux256.jpg -crop 256x256@+256x256 parts-moved.jpg

floyd run --gpu --data randomquark/datasets/bayeux-256-tfrecords/1:customimages "python train.py"
