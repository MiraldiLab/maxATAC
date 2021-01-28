# Training

To train a maxATAC model you need to set up a lot of inputs and a meta table to organize those input files.

___

## Requirements

### Meta Table

The large number of examples, targets, inputs, and peaks are tracked using a meta file. The meta file should be in the following format: 

| Cell_Line | TF   | Output type                        | Experiment date released | File accession | priority | CHIP_Peaks                                                                               | ATAC_Peaks                                                                                   | ATAC_Signal_File                                                                                      | Binding_File                                                                            | Peak_Counts | tuple      | Train_Test_Label |
|-----------|------|------------------------------------|--------------------------|----------------|----------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|-------------|------------|------------------|
| A549      | CTCF | conservative IDR thresholded peaks | 2012-08-20               | ENCFF277ZAR    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/A549__CTCF.bed    | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/A549_ATAC.bed    | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/A549_RPM_minmax01.bw    | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/A549__CTCF.bw    | 36415       | CTCF_36415 | Train            |
| GM12878   | CTCF | conservative IDR thresholded peaks | 2011-02-10               | ENCFF017XLW    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/GM12878__CTCF.bed | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/GM12878_ATAC.bed | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/GM12878_RPM_minmax01.bw | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/GM12878__CTCF.bw | 39892       | CTCF_39892 | Train            |
| HCT116    | CTCF | conservative IDR thresholded peaks | 2012-01-17               | ENCFF832GBA    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HCT116__CTCF.bed  | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/HCT116_ATAC.bed  | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/HCT116_RPM_minmax01.bw  | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HCT116__CTCF.bw  | 49964       | CTCF_49964 | Train            |
| HepG2     | CTCF | conservative IDR thresholded peaks | 2011-03-17               | ENCFF704ECS    | 1        | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HepG2__CTCF.bed   | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_peaks/HepG2_ATAC.bed   | /Users/caz3so/scratch/20201127_maxATAC_refined_samples/ATAC/cell_type_average/HepG2_RPM_minmax01.bw   | /Users/caz3so/scratch/maxATAC/data/20201215_ENCODE_refined_CHIP_rename/HepG2__CTCF.bw   | 44930       | CTCF_44930 | Train            |



You will need to have ATAC-seq and ChIP-seq data in a bigwig format. You will also need peak file for both ATAC-seq and ChIP-seq. If no ATAC-seq or ChIP-seq files are used then you will get an error when building the ROI based training regions. 

___

## run_training

The main function of the training module is the `run_training` function.

The first step of training a maxATAC model is to initialize the Keras model with the architecture of interest. 


___

### get_dilated_cnn

The get_dilated_cnn function takes as input the following 
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

