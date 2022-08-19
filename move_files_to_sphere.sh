#!/bin/bash
for i in {1..20}
do
	echo "Folder ${i}"
	rsync -r --progress imgs/curv_contour_length_18/imgs/$i /sphere/desalab/projects/contour_integration/pathfinder_full/curv_contour_length_18_1M/curv_contour_length_18/
	rsync -r --progress imgs/curv_contour_length_18_neg/imgs/$i /sphere/desalab/projects/contour_integration/pathfinder_full/curv_contour_length_18_1M/curv_contour_length_18_neg/
done
