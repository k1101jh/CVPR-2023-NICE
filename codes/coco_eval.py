# https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py
# https://github.com/salaniz/pycocoevalcap/blob/master/example/coco_eval_example.py

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

annotation_file = 'sample_results/captions_val2014.json'
results_file = 'sample_results/captions_val2014_fakecap_results.json'

# fakecap_results.json은 다음과 같은 형식
# [{"image_id": 391895, "caption": "Man riding a motor bike on a dirt road on the countryside."}, ... ]

# create coco object and coco_result object
coco = COCO(annotation_file)
coco_result = coco.loadRes(results_file)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')