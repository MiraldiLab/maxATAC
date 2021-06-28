import logging
import sys
import os
import glob
import pandas as pd

'''
This chunk of code intends to find the best epoch trained from maxATAC
based on validation dice coeffecient performance for binary models and
based on validation R-squared performance for quantitative models
'''


def find_epoch(args):
    for file in glob.glob(args.model_dir+"/*csv"):
        df = pd.read_csv(file, sep = ',')
            
        if args.quant==False:
            epoch=df['val_dice_coef'].idxmax()
            epoch=epoch+1
            
            logging.error("Best Epoch for " + args.train_tf +" is: " + str(epoch))

        else:
            epoch=df['val_coeff_determination'].idxmax()
            epoch=epoch+1
            
            logging.error("Best Epoch for " + args.train_tf + " is: " + str(epoch))
            

    for model in glob.glob(args.model_dir +"/*"+str(epoch)+".h5"):
        best_model_path=model
        out=pd.DataFrame([[best_model_path]],columns=['Out_Model_Path'])

        out.to_csv(args.model_dir + "/" + args.train_tf + "_best_epoch_path.txt", sep = '\t', index=None, header=None)
    
    sys.exit()

    
    