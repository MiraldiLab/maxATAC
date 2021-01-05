import matplotlib.pyplot as plt
from keras.utils import plot_model

from maxatac.utilities.system_tools import replace_extension, remove_tags


def export_model_structure(model, file_location, suffix="_model_structure", ext=".pdf", skip_tags="_{epoch}"):
    plot_model(
        model=model,
        show_shapes=True,
        show_layer_names=True,
        to_file=replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        )
    )


def export_model_loss(history, file_location, suffix="_model_loss", ext=".pdf", style="ggplot", log_base=10, skip_tags="_{epoch}"):
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

def export_model_dice(history, file_location, suffix="_model_dice", ext=".pdf", style="ggplot", log_base=10, skip_tags="_{epoch}"):
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


def export_model_accuracy(history, file_location, suffix="_model_accuracy", ext=".pdf", style="ggplot", log_base=10, skip_tags="_{epoch}"):
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
