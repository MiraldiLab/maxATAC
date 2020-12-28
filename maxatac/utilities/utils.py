def get_absolute_path(p, cwd_abs_path=None):
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    return p if path.isabs(p) else path.normpath(path.join(cwd_abs_path, p))

def get_dir(dir, permissions=0o0775, exist_ok=True):
    abs_dir = get_absolute_path(dir)
    try:
        makedirs(abs_dir, mode=permissions)
    except error:
        if not exist_ok:
            raise
    return abs_dir

def replace_extension(l, ext):
    return get_absolute_path(
        path.join(
            path.dirname(l),
            get_rootname(l) + ext
        )
    )

def remove_tags(l, tags):
    tags = tags if type(tags) is list else [tags]
    for tag in tags:
        l = l.replace(tag, "")
    return l

class TrainModel(object):
    """
    This is a class for training a maxATAC model

    Args:
        seed (int, optional): Random seed to use.
        out_dir (str): Path to directory for storing results.
        prefix (str): Prefix string for building model name
        arch (str): Architecture to use

    Attributes:
        seed (int): Random state seed.
        out_dir (str): Output directory for storing results.
        model_filename (str): The model filename
        results_location (str): Output directory and model filename
        log_location (str): Path to save logs
        tensor_board_log_dir (str): Path to tensor board log
    """
    def __init__(self, 
                 seed, 
                 out_dir, 
                 prefix
                 ):
        self.seed = random.seed(seed)
        self.out_dir = get_dir(out_dir)
        self.model_filename = prefix + "_{epoch}" + ".h5"
        self.results_location = path.join(self.out_dir, self.model_filename)
        self.log_location = replace_extension(remove_tags(self.results_location, "_{epoch}"), ".csv")
        self.tensor_board_log_dir = get_dir(path.join(self.out_dir, "tensorboard"))

        # Set fit_generator to handle threads by itself
        configure_session(1)

    def InitializeModel (self, 
                         arch,
                         FilterNumber, 
                         KernelSize, 
                         LRate, 
                         decay, 
                         FilterScalingFactor):
        a
        if arch == "DCNN_V2":
            self.nn_model = get_dilated_cnn( input_filters=args.FILTER_NUMBER,
                                        input_kernel_size=args.KERNEL_SIZE,
                                        adam_learning_rate=args.lrate,
                                        adam_decay=args.decay,
                                        filters_scaling_factor=args.FILTERS_SCALING_FACTOR                                 
                                    )

        else:
            sys.exit("Model Architecture not specified correctly. Please check")
        