import os
import numpy as np
import json
from PIL import Image, ImageDraw

def create_visualization(save_loc):
	'''
	Create visualizations of the bounding boxes on red light images
	and save them in the specified directory.
	save_loc specifies where to save visualizations.
	'''

	with open('./data/hw02_preds/preds_test.json') as f:
		data = f.read()

	with open('./data/hw02_annotations/annotations_test.json') as f:
		data_gt = f.read()
		
	# Predicted boxes and ground truth img to box dictionaries
	img_to_box_pred = json.loads(data)
	img_to_box_gt = json.loads(data_gt)

	data_path = './data/RedLights2011_Medium'

	for img in img_to_box_pred:
		# Predicted boxes and ground truth boxes
		pred_boxes = img_to_box_pred[img]
		gt_boxes = img_to_box_gt[img]

		with Image.open(os.path.join(data_path, img)) as im:
			draw = ImageDraw.Draw(im)			
			
			# Draw all predicted boxes in white
			for box in pred_boxes:
				new_box = [box[1], box[0], box[3], box[2]]
				draw.rectangle(new_box, outline='white', width=2)
			
			# Draw all ground truth boxes in green
			for box in gt_boxes:
				new_box = [box[1], box[0], box[3], box[2]]
				draw.rectangle(new_box, outline='green', width=1)

			f_name = save_loc + img.split('.')[0] + '_boxed.jpg'
			im.save(f_name, 'JPEG')

create_visualization('./data/boxed_images/')