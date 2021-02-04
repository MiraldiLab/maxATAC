import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pdb

from maxatac.utilities.helpers import replace_extension, remove_tags


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


def export_loss_dice_accuracy(history, tf, TCL, RR, ARC, file_location, suffix="_model_dice_acc", ext=".png", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False, figsize = (24,12))
    fig.suptitle('Training and Validation: Loss, Dice Coefficient and Accuracy for PCPC training on ' + TCL + '\n' +
                ' with random ratio set at ' + str(RR) + '  for ' + tf + ' ' + ARC + ' architecture'  '\n \n', fontsize=24)
    ###
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

    ###
    t_y = history.history['dice_coef']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_dice_coef"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax2.plot(t_x, t_y, marker='o')
    ax2.plot(v_x, v_y, marker='o')

    ax2.set_xticks(t_x) 
    ax2.set_ylim([0, 1])

    ax2.set_title("Model Dice Coefficient")
    ax2.set_ylabel("Dice Coefficient")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Training", "Validation"], loc="upper left")
    
    ###
    t_y = history.history['acc']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_acc"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax3.plot(t_x, t_y, marker='o')
    ax3.plot(v_x, v_y, marker='o')

    ax3.set_xticks(t_x)
    ax3.set_ylim([0, 1])


    ax3.set_title("Model Accuracy")
    ax3.set_ylabel("Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.legend(["Training", "Validation"], loc="upper left")
    
    fig.tight_layout(pad=10)


    fig.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )


def export_model_loss(history, file_location, suffix="_model_loss", ext=".png", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    
    #t_y = np.log(history.history["loss"]) / np.log(log_base)
    t_y = history.history['loss']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    #v_y = np.log(history.history["val_loss"]) / np.log(log_base)
    v_y = history.history["val_loss"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    plt.plot(t_x, t_y, marker='o')
    plt.plot(v_x, v_y, marker='o')

    plt.xticks(t_x)

    plt.title("Model loss")
    #plt.ylabel(r"$log_{" + str(log_base) + "}Loss$")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper right")

    plt.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )

    plt.close("all")

def export_model_dice(history, file_location, suffix="_model_dice", ext=".png", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    
    t_y = history.history['dice_coef']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_dice_coef"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    plt.plot(t_x, t_y, marker='o')
    plt.plot(v_x, v_y, marker='o')

    plt.xticks(t_x)
    plt.ylim(0, 1)

    plt.title("Model Dice Coefficient")
    plt.ylabel("Dice Coefficient")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")

    plt.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )

    plt.close("all")


def export_model_accuracy(history, file_location, suffix="_model_accuracy", ext=".png", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    
    t_y = history.history['acc']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_acc"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    plt.plot(t_x, t_y, marker='o')
    plt.plot(v_x, v_y, marker='o')

    plt.xticks(t_x)
    plt.ylim(0, 1)

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")

    plt.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )

    plt.close("all")

def export_boxplot(data, file_location, title="Quantile Normalization", names=None, suffix="_boxplot",  ext=".png", style="ggplot"):

    plt.style.use(style)
    
    plt.boxplot(data, showfliers=False)
    
    if names is not None:
        plt.xticks(range(1, len(names) + 1), names, rotation=90)

    plt.title(title)

    plt.savefig(
        replace_extension(file_location, suffix + ext),
        bbox_inches="tight"
    )

    plt.close("all")
    
def export_loss_mse_coeff(history, tf, TCL, RR, ARC, file_location, suffix="_model_loss_mse_coeff", ext=".png", style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, sharex=False, sharey=False, figsize = (24,12))
    fig.suptitle('Training and Validation: Loss, Mean Squared Error and Coefficiennt of Determination for PCPC training on ' + TCL + '\n' +
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
    #ax3.set_ylim([0, 1])


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

    
    fig.tight_layout(pad=10)


    fig.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )


def export_prc(precision, recall, file_location,  title="Precision Recall Curve", suffix="_prc", ext=".png", style="ggplot"):
    
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