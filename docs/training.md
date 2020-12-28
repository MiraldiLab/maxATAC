# Training

## run_training

The main function of the training module is the `run_training` function.

Input: The args namespace object with the following attributes

* args.arch
* args.seed
* args.prefix
* args.output
* args.KERNEL_SIZE
* args.FILTER_NUMBER
* args.lrate
* args.decay
* args.weights

___

### get_dilated_cnn

The get_dilated_cnn function takes as input the following parameters defined by the arg parser and constants file:

* input_filters,
* input_kernel_size,
* adam_learning_rate,
* adam_decay,
* input_length=INPUT_LENGTH,
* input_channels=INPUT_CHANNELS,
* input_activation=INPUT_ACTIVATION,
* output_filters=OUTPUT_FILTERS,
* output_kernel_size=OUTPUT_KERNEL_SIZE,
* output_activation=OUTPUT_ACTIVATION,
* filters_scaling_factor=FILTERS_SCALING_FACTOR,
* conv_blocks=CONV_BLOCKS,
* padding=PADDING,
* pool_size=POOL_SIZE,
* adam_beta_1=ADAM_BETA_1,
* adam_beta_2=ADAM_BETA_2,
* dilation_rate=DILATION_RATE,                                  
* weights=None

The function get_dilated_cnn will build the CNN model.

The first part of building the CNN model is to set up the input layer and the temporary variables used for the input layer and filters. The input layer is the shape of the input lengths (1024 bp) and the input channels (5;4 DNA + ATAC).

<pre>
# Inputs
input_layer = Input(shape=(input_length, input_channels))

# Temporary variables redefined in encoder/decoder loops
layer = input_layer  

filters = input_filters  
</pre>

The next part is to set up the encoder. Our models has 6 convolutional blocks. We will loop through each block and set up the layers individually. The index of the conv_blocks is used to find the dilation rate for the layer. 
The dilation rates are 1,1,2,4,8,16. We also use the "same" padding for our 1D convolutions. Reference this site for more information: [Padding Examples](https://www.machinecurve.com/index.php/2020/02/07/what-is-padding-in-a-neural-network/)

The number of filters increases with each layer due to the idea that simpler representations combine to make higher order combinations. Our current method of choosing these filters per layer is based on the Leopard implementation of the algorithm. 
The number of filters: 15, 22, 33, 50, 75

I was thinking we might double each filter or triple them. If we double the number of filters we end up with the following sequence:
15, 30, 60, 120, 240

If we triple them we end up with the following sequence:
15, 45, 135, 405, 1215

<pre>
# Encoder
all_layers = []

for i in range(conv_blocks-1):          # [0, 1, 2, 3, 4, 5]
    layer_dilation_rate = dilation_rate[i] # [1, 1, 2, 4, 8, 16]
    
    layer = get_layer(
                inbound_layer=layer, # input_layer is used wo MaxPooling1D
                filters=filters,
                kernel_size=input_kernel_size,
                activation=input_activation,
                padding=padding,
                dilation_rate= layer_dilation_rate
    )

    if i < conv_blocks - 1:  # need to update all except the last layers
        filters = round(filters * filters_scaling_factor)
        layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)(layer)
    
    all_layers.append(layer)

# Outputs
layer_dilation_rate = dilation_rate[-1]

output_layer = get_layer(
    inbound_layer=layer,
    filters=output_filters,
    kernel_size=output_kernel_size,
    activation=output_activation,
    padding=padding,
    dilation_rate=layer_dilation_rate,
    skip_batch_norm=True,
    n=1
)
newdim = tuple([x for x in output_layer.shape.as_list() if x != 1 and x is not None])

output_layer = Reshape(newdim)(output_layer)
</pre>

The batch normalization refers to whether the scores are normalized between 0,1. The scores from that activation function can be any real value so those are transformed into something that is comparable.

The last step of building the model is to compile the model based on the learning rate that is desired. You should also set up the metrics that are used to evaluate the model. 
<pre>
# Model
model = Model(inputs=[input_layer], outputs=[output_layer])

model.compile(
    optimizer=Adam(
        lr=adam_learning_rate,
        beta_1=adam_beta_1,
        beta_2=adam_beta_2,
        decay=adam_decay
    ),
    loss=loss_function,
    metrics=[dice_coef, 'accuracy']
)
</pre>

___

#### get_layer

The get_layer function returns new layer without max pooling. If concat_layer, transpose_kernel_size and transpose_strides are provided run Conv1DTranspose and Concatenation. Optionally, you can skip batch normalization. The inputs are:

* inbound_layer,
* filters,
* kernel_size,
* activation,
* padding,
* dilation_rate=1,
* skip_batch_norm=False,
* concat_layer=None,
* transpose_kernel_size=None,
* transpose_strides=None,
* n=2

<pre>
for i in range(n):
    inbound_layer = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        dilation_rate=dilation_rate
    )(inbound_layer)

    if not skip_batch_norm:
        inbound_layer = BatchNormalization()(inbound_layer)

return inbound_layer
</pre>

___

### get_roi_pool


<pre>
def get_roi_pool(seq_len=None, roi=None, shuffle=False):
    roi_df = pd.read_csv(roi, sep="\t", header=0, index_col=None)
    temp = roi_df['Stop'] - roi_df['Start']
    ##############################
    #Temporary Workaround. Needs to be deleted later 
    roi_ok = (temp == seq_len)
    temp_df = roi_df[roi_ok==True]
    roi_df = temp_df
    ###############################

    #roi_ok = (temp == seq_len).all()
    #if not roi_ok:
        
        #sys.exit("ROI Length Does Not Match Input Length")
        
    if shuffle:
        roi_df = roi_df.sample(frac=1)
    return roi_df
</pre>

___

### validate_pool

