# NJM
### Basic Usage
Reference: https://github.com/NJMCODE2018/NJM

#### Example
You can change the train config in 'config/'
Please first run NJM_train.ini and train at least 1 epoch to generate dataset from raw data
execute the following command from the project home directory:<br/>
	``python run.py --config_pth NJM_train.ini``
<br/>debug:<br/>
	``python run.py --config_pth NJM_debug.ini``

## Environment Settings
- Python version:  '3.6'
- torch version: '1.10.2+cu113'
