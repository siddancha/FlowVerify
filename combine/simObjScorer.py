import os
import json
import numpy as np
import argparse

from flowmatch.utils import load_config
from pipeline.eval.utils import compute_iou

class SimObjScorer:
	def __init__(self, dt_pth):
		with open(dt_pth) as f:
			self.dt = json.load(f)
		print(len(self.dt))
		

	def create_score_files(self, out_dir, iou=0.5, nvp=15):
		self.scores = self._get_SimObj_scores(self.dt, iou, nvp)
		with open(os.path.join(out_dir, 'SimObj_{}.json'.format(iou)), 'w') as f:
			json.dump(self.scores, f)

	def _get_SimObj_scores(self, dt, iou, nvp):
		'''
		dt: detections are grouped/sorted by image_id
		'''
		if len(dt) == 0:
			return []
		image_id, boxes = dt[0]['image_id'], []
		res = []
		img_ids = []
		for js in dt:
			img_id = js['image_id']
			if img_id == image_id:
				boxes.append(js)
			else:
				scene_scores = self._scene_SimObj(boxes, iou, nvp)
				boxes = [js]
				image_id = img_id
				res += scene_scores
		res += self._scene_SimObj(boxes, iou, nvp)
		print(len(res))
		return res

	def _scene_SimObj(self, detections, iou, nvp):
		detections.sort(key=lambda x: x['score'], reverse=True)
		scores = [x['score'] for x in detections]

		res = {x['id']: 0 for x in detections}
		boxes = np.array([ x['bbox'] for x in detections])
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]+x1
		y2 = boxes[:, 3]+y1
		areas = (x2 - x1 + 1) * (y2 - y1 + 1)

		for i in range(len(detections)):
			other_x1s, other_y1s, other_x2s, other_y2s, other_areas, other_scores, other_dets  =\
				self._others([x1, y1, x2, y2, areas, scores, detections], i)

			xx1 = np.maximum(x1[i], other_x1s)
			yy1 = np.maximum(y1[i], other_y1s)
			xx2 = np.minimum(x2[i], other_x2s)
			yy2 = np.minimum(y2[i], other_y2s)

			w = np.maximum(0.0, xx2 - xx1 + 1)
			h = np.maximum(0.0, yy2 - yy1 + 1)
			inter = w * h

			ovr = inter / (areas[i] + other_areas - inter)
			over_ind = np.where(ovr > iou)[0]
			cid = detections[i]['category_id']
			over_inds = [x for x in over_ind if other_dets[x]['category_id'] != cid]
			if not len(over_inds) == len(over_ind):
				for i in over_ind:
					print(ovr[i], [x[i] for x in [other_x1s, other_y1s, other_x2s, other_y2s]])
					print(detections[i]['bbox'])
			assert(len(over_inds) == len(over_ind))
			if len(over_inds) == 0:
				res[detections[i]['id']] = detections[i]['score']
			else:
				next_highest_score = other_scores[over_inds[0]]
				res[detections[i]['id']] = max(0, detections[i]['score'] - next_highest_score)

		res = [{'id': x, 'scores': [res[x]] * nvp} for x in res]
		return res

	def _others(self, arrs, i):
		return [np.concatenate([arr[:i], arr[i+1:]]) for arr in arrs]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='config.yaml', help='config file path')
	parser.add_argument('--eval_ds', default='gmu_test', help='dataset to generate SimObj scores')
	parser.set_defaults()

	args = parser.parse_args()

	eval_ds = args.eval_ds
	cfg = load_config(args.config).combine

	scorer = SimObjScorer(os.path.join(cfg.root, cfg[eval_ds].dt_json))
	scorer.create_score_files(os.path.join(cfg.root, cfg[eval_ds].run_dir))


