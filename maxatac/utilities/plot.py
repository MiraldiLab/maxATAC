from maxatac.utilities.system_tools import replace_extension, remove_tags, Mute
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig

with Mute():
    from tensorflow.keras.utils import plot_model


def export_model_structure(model, file_location, suffix="_model_structure", ext=".png", skip_tags="_{epoch}"):
    plot_model(
        model=model,
        show_shapes=True,
        show_layer_names=True,
        to_file=replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        )
    )


def export_binary_metrics(history, tf, RR, ARC, file_location, best_epoch, suffix="_model_dice_acc", ext=".png",
                          style="seaborn_v0_8-whitegrid", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(18, 6), dpi=200)
    fig.suptitle('Loss and Dice Coefficient with random ratio set at ' + str(RR) + '  for ' + tf + ' using the ' + ARC + ' architecture', fontsize=22)

    ### Loss
    t_y = history.history['loss']
    v_y = history.history["val_loss"]

    x = [int(i) for i in range(1, len(t_y) + 1)]

    ax1.plot(x, t_y, marker='o', lw=3)
    ax1.plot(x, v_y, marker='o', lw=3)

    ax1.set_title("Model Loss", size=20)
    ax1.set_ylabel("Loss", size=20)
    ax1.set_xlabel("Epoch", size=20)

    ax1.legend(["Training", "Validation"], loc="upper right")

    ax1.set_xticks(x)
    ax1.set_xlim(0, )
    ax1.set_ylim(0, )
    ax1.tick_params(labelsize=16)
    ax1.axvline(best_epoch, lw=3, color="red")

    ax1.tick_params(axis="y", labelsize=16)
    ax1.tick_params(axis="x", labelsize=12, rotation=90)

    ### Dice Coefficient
    t_y = history.history['dice_coef']
    v_y = history.history["val_dice_coef"]

    ax2.plot(x, t_y, marker='o', lw=3)
    ax2.plot(x, v_y, marker='o', lw=3)

    ax2.set_title("Model Dice Coefficient", size=20)
    ax2.set_ylabel("Dice Coefficient", size=20)
    ax2.set_xlabel("Epoch", size=20)

    ax2.legend(["Training", "Validation"], loc="upper left")

    ax2.set_xticks(x)
    ax2.set_xlim(0, )
    ax2.set_ylim(0, )

    ax2.axvline(best_epoch, lw=3, color="red")

    ax2.tick_params(axis="y", labelsize=16)
    ax2.tick_params(axis="x", labelsize=12, rotation=90)

    fig.tight_layout()

    fig.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )


def export_loss_mse_coeff(history, tf, TCL, RR, ARC, file_location, suffix="_model_loss_mse_coeff", ext=".png",
                          style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    fig, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(24, 12))
    fig.suptitle(
        'Training and Validation: Loss, Mean Squared Error and Coefficiennt of Determination for PCPC training on ' + TCL + '\n' +
        ' with random ratio set at ' + str(RR) + '  for ' + tf + ' ' + ARC + ' architecture'  '\n \n', fontsize=24)

    t_y = history.history['loss']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_loss"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax1.plot(t_x, t_y, marker='o')
    ax1.plot(v_x, v_y, marker='o')

    ax1.set_xticks(t_x)

    ax1.set_title("Model loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Training", "Validation"], loc="upper right")

    t_y = history.history['loss']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_loss"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax2.plot(t_x, t_y, marker='o')
    ax2.plot(v_x, v_y, marker='o')

    ax2.set_xticks(t_x)
    ax2.set_yscale('log')

    ax2.set_title("Model Log scale loss")
    ax2.set_ylabel("Loss Log-Scale")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Training", "Validation"], loc="upper right")

    t_y = history.history['coeff_determination']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_coeff_determination"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax3.plot(t_x, t_y, marker='o')
    ax3.plot(v_x, v_y, marker='o')

    ax3.set_xticks(t_x)
    # ax3.set_ylim([0, 1])

    ax3.set_title("R Squared")
    ax3.set_ylabel("R Squared")
    ax3.set_xlabel("Epoch")
    ax3.legend(["Training", "Validation"], loc="upper left")

    t_y = history.history['coeff_determination']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_coeff_determination"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax4.plot(t_x, t_y, marker='o')
    ax4.plot(v_x, v_y, marker='o')

    ax4.set_xticks(t_x)
    ax4.set_ylim([0, 1])

    ax4.set_title("R Squared")
    ax4.set_ylabel("R Squared")
    ax4.set_xlabel("Epoch")
    ax4.legend(["Training", "Validation"], loc="upper left")

    t_y = history.history['precision']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_precision"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax5.plot(t_x, t_y, marker='o')
    ax5.plot(v_x, v_y, marker='o')
    ax5.set_xticks(t_x)
    ax5.set_ylim([0, 1])

    ax5.set_title("Precision")
    ax5.set_ylabel("Precision")
    ax5.set_xlabel("Epoch")
    ax5.legend(["Training", "Validation"], loc="upper left")

    #
    t_y = history.history['recall']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_recall"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax6.plot(t_x, t_y, marker='o')
    ax6.plot(v_x, v_y, marker='o')

    ax6.set_xticks(t_x)
    ax6.set_ylim([0, 1])

    ax6.set_title("Recall")
    ax6.set_ylabel("Recall")
    ax6.set_xlabel("Epoch")
    ax6.legend(["Training", "Validation"], loc="upper left")
    #
    t_y = history.history['pearson']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_pearson"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax7.plot(t_x, t_y, marker='o')
    ax7.plot(v_x, v_y, marker='o')

    ax7.set_xticks(t_x)
    ax7.set_ylim([0, 1])

    ax7.set_title("Pearson Correlation")
    ax7.set_ylabel("Pearson Correlation")
    ax7.set_xlabel("Epoch")
    ax7.legend(["Training", "Validation"], loc="upper left")
    #
    t_y = history.history['spearman']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_spearman"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax8.plot(t_x, t_y, marker='o')
    ax8.plot(v_x, v_y, marker='o')

    ax8.set_xticks(t_x)
    ax8.set_ylim([0, 1])

    ax8.set_title("Spearman Correlation")
    ax8.set_ylabel("Spearman Correlation")
    ax8.set_xlabel("Epoch")
    ax8.legend(["Training", "Validation"], loc="upper left")

    fig.tight_layout(pad=10)

    fig.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )


def export_prc(precision, recall, file_location, title="Precision Recall Curve", suffix="_prc", ext=".png",
               style="ggplot"):
    plt.style.use(style)

    plt.plot(recall, precision)

    plt.title(title)
    plt.ylabel("Precision")
    plt.xlabel("Recall")

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.savefig(
        replace_extension(file_location, suffix + ext),
        bbox_inches="tight"
    )

    plt.close("all")


def plot_chromosome_scores_dist(input_bigwig, chrom_name, region_start, region_stop):
    with pyBigWig.open(input_bigwig) as input_bw:
        chr_vals = input_bw.values(chrom_name, region_start, region_stop, numpy=True)

    plt.hist(chr_vals[chr_vals > 0])

# Create a multi-paneled plot to display the threshold calibration statistics (as seen in the maxATAC_data repository).
def plot_threshold_calibration_stats(thresholds, precision, recall, log2fc, f1, file_location, prefix, 
                                     suffix = "_validationPerformance_vs_thresholdCalibration", ext = ".png", style = "ggplot"):
    plt.style.use(style)
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 12))
    
    # Subfigure 1: validation precision v. threshold values
    axs[0, 0].plot(thresholds, precision, c = "black", lw = 3)
    axs[0, 0].set_title("chr2 Validation Precision v. Thresholds", size = "medium")
    axs[0, 0].set_xlabel("Threshold", size = "medium")
    axs[0, 0].set_xlim([0.0, 1.0])
    axs[0, 0].set_ylabel("Validation Precision", size = "medium")
    axs[0, 0].legend(["Monotonic Median Precision"], loc = "upper left")
    
    # Subfigure 2: validation log2(FC) v. threshold values
    axs[0, 1].plot(thresholds, log2fc, c = "black", lw = 3)
    axs[0, 1].set_title("chr2 Validation log2(FC) v. Thresholds", size = "medium")
    axs[0, 1].set_xlabel("Threshold", size = "medium")
    axs[0, 1].set_xlim([0.0, 1.0])
    axs[0, 1].set_ylabel("Validation log2(FC)", size = "medium")
    axs[0, 1].legend(["Monotonic Median log2(FC)"], loc = "lower left")
    
    # Subfigure 3: validation recall v. threshold values
    axs[1, 0].plot(thresholds, recall, c = "black", lw = 3)
    axs[1, 0].set_title("chr2 Validation Recall v. Thresholds", size = "medium")
    axs[1, 0].set_xlabel("Threshold", size = "medium")
    axs[1, 0].set_xlim([0.0, 1.0])
    axs[1, 0].set_ylabel("Validation Recall", size = "medium")
    axs[1, 0].legend(["Monotonic Median Recall"], loc = "upper right")
    
    # Subfigure 4: validation F1 score v. threshold values
    axs[1, 1].plot(thresholds, f1, c = "black", lw = 3)
    axs[1, 1].set_title("chr2 Validation F1 Score v. Thresholds", size = "medium")
    axs[1, 1].set_xlabel("Threshold", size = "medium")
    axs[1, 1].set_xlim([0.0, 1.0])
    axs[1, 1].set_ylabel("Validation F1 Score", size = "medium")
    axs[1, 1].legend(["Median F1 Score"], loc = "upper right")

    # Save the figure.
    fig.suptitle(prefix + " chr2 Validation Performance v. Threshold Calibration")
    fig.savefig(replace_extension(file_location, prefix + '_' + suffix + ext), bbox_inches = "tight", dpi = 320)
