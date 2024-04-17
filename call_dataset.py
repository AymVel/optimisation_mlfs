#from dataset.azure import get_azure, merge_dataset
#from dataset.ai4i import get_ai4i
from dataset.get_hydraulic import load_Hydraulic

import warnings
warnings.filterwarnings('ignore')




def select_dataset(ds_name,cmd=False):

    '''if(ds_name == "azure"):
        X_train, X_test, y_train, y_test, class_name = merge_dataset(already_loaded=True,cmd=cmd)#get_azure(period=24)
        #print(f"AZURE: X_train: {X_train.shape} | X_test: {X_test.shape} | Y_test: {y_test.shape}")

    if(ds_name == "AI4I*"):
        X_train, X_test, y_train, y_test, class_name = get_ai4i(with_machine_failure=False,cmd=cmd)  # Get dataset already splitted

    if (ds_name == "AI4I**"):
        X_train, X_test, y_train, y_test, class_name = get_ai4i( with_machine_failure=True,cmd=cmd)  # Get dataset already splitted
    '''
    if(ds_name == "hydraulic"):
        X_train, X_test, y_train, y_test, class_name = load_Hydraulic(encoding="binary",cmd=cmd)  # Get dataset already splitted

    return X_train, X_test, y_train, y_test, class_name


ds_name = ["hydraulic"]#["hydraulic","AI4I*","AI4I**"]#"azure"]
cmd = True # when called from root project folder
for ds in ds_name:
    print(f"Dataset = {ds}")
    X_train, X_test, y_train, y_test, class_name = select_dataset(ds_name=ds,cmd=True)