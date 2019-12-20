# FlowVerify
Source code for "Combining Deep Learning and Verification forPrecise Object Instance Detection"

### Usage:
#### To setup this repo:
cd FlowVerify

export PYTHONPATH=/path/to/FlowVerify

#### To train FlowMatchNet:
cd FlowVerify/flowmatch/exps/coco

python main.py --config=fn_cc/whole/config

#### To run FlowMatchNet on detector outputs and generate score files:
cd FlowVerify/pipeline/eval

python run_flowverify.py --config=../configs/tdid_rgbd.yaml --model-path=/path/to/pretrained/model --out-dir=/path/to/output/dir --save-every=100

Check files in FlowVerify/pipeline/configs/ for example configuration files

#### To evaluate using FlowVerify:
cd FlowVerify/combine

python eval_params.py --config=config.yaml --eval_ds=gmu_test

Make sure to change paths in FlowVerify/combine/config.yaml to point to desired files
