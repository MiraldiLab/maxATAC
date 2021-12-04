import numpy as np
import pandas as pd
import os
import multiprocessing as m
from multiprocessing import Pool
import time
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank
from sklearn.metrics import precision_recall_curve
from sklearn import metrics


def extract_pred_gs_bw(bigwig_file, training_data_dict, chrom_name, chrom_length, bin_count):
    start = time.time()
    bw_name = bigwig_file.split("/")[6]
    chrom_vals = get_bigwig_stats(bigwig_file, chrom_name, chrom_length, bin_count)
    print(training_data_dict[bigwig_file])
    
    
    predictions = np.empty(1)
    gold_standard = np.empty(1)
 
    goldstandard_array = import_GoldStandard_array(training_data_dict[bigwig_file], chrom_name, chrom_length, bin_count)
    
    predictions = np.concatenate([predictions, chrom_vals])
    gold_standard = np.concatenate([gold_standard, goldstandard_array])
    
    end = time.time()
    print('total time (s)= ' + str(end-start), "____________", bw_name)
    
    return predictions, gold_standard

def run_thresholding(args):
    """
    :param args:
    :return:
    """
    # Make the output directory
    output_dir = get_dir(args.output_dir)

    chromosome_sizes_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    meta_DF = pd.read_table(args.meta_file)

    training_data_dict = pd.Series(meta_DF["Binding_File"].values,index=meta_DF["Prediction"]).to_dict()

    results_filename = os.path.join(output_dir,
                                    args.prefix + "_" + args.chromosomes[0] + "_" + str(args.bin_size) + "bp_PRC.tsv")

    # Loop through the chromosomes and average the values across files
    OUT=[]    

    for chrom_name, chrom_length in chromosome_sizes_dictionary.items():
        bin_count = int(int(chrom_length) / int(args.bin_size))  # need to floor the number
        
        blacklist_mask = import_blacklist_mask(args.blacklist, chrom_name, chrom_length, bin_count)
        
        blacklist = np.repeat(blacklist_mask, len(training_data_dict.keys()))
        
        lst_of_bws=list(training_data_dict.keys())

        pool = Pool(8) 
        output = pool.starmap(
            extract_pred_gs_bw,
            [(bigwig, training_data_dict, chrom_name, chrom_length, bin_count) for bigwig in lst_of_bws]
                            )
        OUT.append(output)
        #if __name__ == '__main__':
        
        '''pool = Pool(8)                         
        output = pool.map(extract_pred_gs_bw, lst)
        OUT.append(output)'''
 
    
    DF=pd.DataFrame([])
    for i in range(len(OUT[0])):
        df = pd.DataFrame([])
        df['Prediction'] = OUT[0][i][0][:bin_count].tolist()
        df['GoldStandard'] = OUT[0][i][1][:bin_count].tolist()
        #df = pd.DataFrame(OUT[0][i]).transpose()
        print(df.shape)
        DF = DF.append(df)
    
    
    DF.loc[DF['GoldStandard'] != 1, 'GoldStandard'] = 0
    #DF.columns = ['Prediction', 'GoldStandard']
    precision, recall, thresholds = precision_recall_curve(DF['GoldStandard'][blacklist], DF['Prediction'][blacklist])
    
    
    # Create a dataframe from the results
    PR_CURVE_DF = pd.DataFrame({'Precision': precision, 'Recall': recall, "Threshold": np.insert(thresholds, 0, 0)})
    
    # Calculate AUPRc
    AUPRC = metrics.auc(y=precision[:-1], x=recall[:-1])

    PR_CURVE_DF["AUC"] = AUPRC
    #remove all rows with a 0
    #DF_new = DF[(DF != 0).all(1)]
    
    PR_CURVE_DF.to_csv(results_filename, sep="\t", header=True, index=False)
    
    
    
    



