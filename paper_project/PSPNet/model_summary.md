### keras resnet50 model:
``` text
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 224, 224, 3)   0                                            
____________________________________________________________________________________________________
zeropadding2d_1 (ZeroPadding2D)  (None, 230, 230, 3)   0           input_1[0][0]                    
____________________________________________________________________________________________________
conv1 (Convolution2D)            (None, 112, 112, 64)  9472        zeropadding2d_1[0][0]            
____________________________________________________________________________________________________
bn_conv1 (BatchNormalization)    (None, 112, 112, 64)  256         conv1[0][0]                      
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 112, 112, 64)  0           bn_conv1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 55, 55, 64)    0           activation_1[0][0]               
____________________________________________________________________________________________________
res2a_branch2a (Convolution2D)   (None, 55, 55, 64)    4160        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
bn2a_branch2a (BatchNormalizatio (None, 55, 55, 64)    256         res2a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 55, 55, 64)    0           bn2a_branch2a[0][0]              
____________________________________________________________________________________________________
res2a_branch2b (Convolution2D)   (None, 55, 55, 64)    36928       activation_2[0][0]               
____________________________________________________________________________________________________
bn2a_branch2b (BatchNormalizatio (None, 55, 55, 64)    256         res2a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 55, 55, 64)    0           bn2a_branch2b[0][0]              
____________________________________________________________________________________________________
res2a_branch2c (Convolution2D)   (None, 55, 55, 256)   16640       activation_3[0][0]               
____________________________________________________________________________________________________
res2a_branch1 (Convolution2D)    (None, 55, 55, 256)   16640       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
bn2a_branch2c (BatchNormalizatio (None, 55, 55, 256)   1024        res2a_branch2c[0][0]             
____________________________________________________________________________________________________
bn2a_branch1 (BatchNormalization (None, 55, 55, 256)   1024        res2a_branch1[0][0]              
____________________________________________________________________________________________________
merge_1 (Merge)                  (None, 55, 55, 256)   0           bn2a_branch2c[0][0]              
                                                                   bn2a_branch1[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 55, 55, 256)   0           merge_1[0][0]                    
____________________________________________________________________________________________________
res2b_branch2a (Convolution2D)   (None, 55, 55, 64)    16448       activation_4[0][0]               
____________________________________________________________________________________________________
bn2b_branch2a (BatchNormalizatio (None, 55, 55, 64)    256         res2b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 55, 55, 64)    0           bn2b_branch2a[0][0]              
____________________________________________________________________________________________________
res2b_branch2b (Convolution2D)   (None, 55, 55, 64)    36928       activation_5[0][0]               
____________________________________________________________________________________________________
bn2b_branch2b (BatchNormalizatio (None, 55, 55, 64)    256         res2b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 55, 55, 64)    0           bn2b_branch2b[0][0]              
____________________________________________________________________________________________________
res2b_branch2c (Convolution2D)   (None, 55, 55, 256)   16640       activation_6[0][0]               
____________________________________________________________________________________________________
bn2b_branch2c (BatchNormalizatio (None, 55, 55, 256)   1024        res2b_branch2c[0][0]             
____________________________________________________________________________________________________
merge_2 (Merge)                  (None, 55, 55, 256)   0           bn2b_branch2c[0][0]              
                                                                   activation_4[0][0]               
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 55, 55, 256)   0           merge_2[0][0]                    
____________________________________________________________________________________________________
res2c_branch2a (Convolution2D)   (None, 55, 55, 64)    16448       activation_7[0][0]               
____________________________________________________________________________________________________
bn2c_branch2a (BatchNormalizatio (None, 55, 55, 64)    256         res2c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 55, 55, 64)    0           bn2c_branch2a[0][0]              
____________________________________________________________________________________________________
res2c_branch2b (Convolution2D)   (None, 55, 55, 64)    36928       activation_8[0][0]               
____________________________________________________________________________________________________
bn2c_branch2b (BatchNormalizatio (None, 55, 55, 64)    256         res2c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 55, 55, 64)    0           bn2c_branch2b[0][0]              
____________________________________________________________________________________________________
res2c_branch2c (Convolution2D)   (None, 55, 55, 256)   16640       activation_9[0][0]               
____________________________________________________________________________________________________
bn2c_branch2c (BatchNormalizatio (None, 55, 55, 256)   1024        res2c_branch2c[0][0]             
____________________________________________________________________________________________________
merge_3 (Merge)                  (None, 55, 55, 256)   0           bn2c_branch2c[0][0]              
                                                                   activation_7[0][0]               
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 55, 55, 256)   0           merge_3[0][0]                    
____________________________________________________________________________________________________
res3a_branch2a (Convolution2D)   (None, 28, 28, 128)   32896       activation_10[0][0]              
____________________________________________________________________________________________________
bn3a_branch2a (BatchNormalizatio (None, 28, 28, 128)   512         res3a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 28, 28, 128)   0           bn3a_branch2a[0][0]              
____________________________________________________________________________________________________
res3a_branch2b (Convolution2D)   (None, 28, 28, 128)   147584      activation_11[0][0]              
____________________________________________________________________________________________________
bn3a_branch2b (BatchNormalizatio (None, 28, 28, 128)   512         res3a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 28, 28, 128)   0           bn3a_branch2b[0][0]              
____________________________________________________________________________________________________
res3a_branch2c (Convolution2D)   (None, 28, 28, 512)   66048       activation_12[0][0]              
____________________________________________________________________________________________________
res3a_branch1 (Convolution2D)    (None, 28, 28, 512)   131584      activation_10[0][0]              
____________________________________________________________________________________________________
bn3a_branch2c (BatchNormalizatio (None, 28, 28, 512)   2048        res3a_branch2c[0][0]             
____________________________________________________________________________________________________
bn3a_branch1 (BatchNormalization (None, 28, 28, 512)   2048        res3a_branch1[0][0]              
____________________________________________________________________________________________________
merge_4 (Merge)                  (None, 28, 28, 512)   0           bn3a_branch2c[0][0]              
                                                                   bn3a_branch1[0][0]               
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 28, 28, 512)   0           merge_4[0][0]                    
____________________________________________________________________________________________________
res3b_branch2a (Convolution2D)   (None, 28, 28, 128)   65664       activation_13[0][0]              
____________________________________________________________________________________________________
bn3b_branch2a (BatchNormalizatio (None, 28, 28, 128)   512         res3b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 28, 28, 128)   0           bn3b_branch2a[0][0]              
____________________________________________________________________________________________________
res3b_branch2b (Convolution2D)   (None, 28, 28, 128)   147584      activation_14[0][0]              
____________________________________________________________________________________________________
bn3b_branch2b (BatchNormalizatio (None, 28, 28, 128)   512         res3b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 28, 28, 128)   0           bn3b_branch2b[0][0]              
____________________________________________________________________________________________________
res3b_branch2c (Convolution2D)   (None, 28, 28, 512)   66048       activation_15[0][0]              
____________________________________________________________________________________________________
bn3b_branch2c (BatchNormalizatio (None, 28, 28, 512)   2048        res3b_branch2c[0][0]             
____________________________________________________________________________________________________
merge_5 (Merge)                  (None, 28, 28, 512)   0           bn3b_branch2c[0][0]              
                                                                   activation_13[0][0]              
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 28, 28, 512)   0           merge_5[0][0]                    
____________________________________________________________________________________________________
res3c_branch2a (Convolution2D)   (None, 28, 28, 128)   65664       activation_16[0][0]              
____________________________________________________________________________________________________
bn3c_branch2a (BatchNormalizatio (None, 28, 28, 128)   512         res3c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 28, 28, 128)   0           bn3c_branch2a[0][0]              
____________________________________________________________________________________________________
res3c_branch2b (Convolution2D)   (None, 28, 28, 128)   147584      activation_17[0][0]              
____________________________________________________________________________________________________
bn3c_branch2b (BatchNormalizatio (None, 28, 28, 128)   512         res3c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_18 (Activation)       (None, 28, 28, 128)   0           bn3c_branch2b[0][0]              
____________________________________________________________________________________________________
res3c_branch2c (Convolution2D)   (None, 28, 28, 512)   66048       activation_18[0][0]              
____________________________________________________________________________________________________
bn3c_branch2c (BatchNormalizatio (None, 28, 28, 512)   2048        res3c_branch2c[0][0]             
____________________________________________________________________________________________________
merge_6 (Merge)                  (None, 28, 28, 512)   0           bn3c_branch2c[0][0]              
                                                                   activation_16[0][0]              
____________________________________________________________________________________________________
activation_19 (Activation)       (None, 28, 28, 512)   0           merge_6[0][0]                    
____________________________________________________________________________________________________
res3d_branch2a (Convolution2D)   (None, 28, 28, 128)   65664       activation_19[0][0]              
____________________________________________________________________________________________________
bn3d_branch2a (BatchNormalizatio (None, 28, 28, 128)   512         res3d_branch2a[0][0]             
____________________________________________________________________________________________________
activation_20 (Activation)       (None, 28, 28, 128)   0           bn3d_branch2a[0][0]              
____________________________________________________________________________________________________
res3d_branch2b (Convolution2D)   (None, 28, 28, 128)   147584      activation_20[0][0]              
____________________________________________________________________________________________________
bn3d_branch2b (BatchNormalizatio (None, 28, 28, 128)   512         res3d_branch2b[0][0]             
____________________________________________________________________________________________________
activation_21 (Activation)       (None, 28, 28, 128)   0           bn3d_branch2b[0][0]              
____________________________________________________________________________________________________
res3d_branch2c (Convolution2D)   (None, 28, 28, 512)   66048       activation_21[0][0]              
____________________________________________________________________________________________________
bn3d_branch2c (BatchNormalizatio (None, 28, 28, 512)   2048        res3d_branch2c[0][0]             
____________________________________________________________________________________________________
merge_7 (Merge)                  (None, 28, 28, 512)   0           bn3d_branch2c[0][0]              
                                                                   activation_19[0][0]              
____________________________________________________________________________________________________
activation_22 (Activation)       (None, 28, 28, 512)   0           merge_7[0][0]    block3的输出，output_stride=8                 
____________________________________________________________________________________________________
res4a_branch2a (Convolution2D)   (None, 14, 14, 256)   131328      activation_22[0][0]              
____________________________________________________________________________________________________
bn4a_branch2a (BatchNormalizatio (None, 14, 14, 256)   1024        res4a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_23 (Activation)       (None, 14, 14, 256)   0           bn4a_branch2a[0][0]              
____________________________________________________________________________________________________
res4a_branch2b (Convolution2D)   (None, 14, 14, 256)   590080      activation_23[0][0]              
____________________________________________________________________________________________________
bn4a_branch2b (BatchNormalizatio (None, 14, 14, 256)   1024        res4a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_24 (Activation)       (None, 14, 14, 256)   0           bn4a_branch2b[0][0]              
____________________________________________________________________________________________________
res4a_branch2c (Convolution2D)   (None, 14, 14, 1024)  263168      activation_24[0][0]              
____________________________________________________________________________________________________
res4a_branch1 (Convolution2D)    (None, 14, 14, 1024)  525312      activation_22[0][0]              
____________________________________________________________________________________________________
bn4a_branch2c (BatchNormalizatio (None, 14, 14, 1024)  4096        res4a_branch2c[0][0]             
____________________________________________________________________________________________________
bn4a_branch1 (BatchNormalization (None, 14, 14, 1024)  4096        res4a_branch1[0][0]              
____________________________________________________________________________________________________
merge_8 (Merge)                  (None, 14, 14, 1024)  0           bn4a_branch2c[0][0]              
                                                                   bn4a_branch1[0][0]               
____________________________________________________________________________________________________
activation_25 (Activation)       (None, 14, 14, 1024)  0           merge_8[0][0]                    
____________________________________________________________________________________________________
res4b_branch2a (Convolution2D)   (None, 14, 14, 256)   262400      activation_25[0][0]              
____________________________________________________________________________________________________
bn4b_branch2a (BatchNormalizatio (None, 14, 14, 256)   1024        res4b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_26 (Activation)       (None, 14, 14, 256)   0           bn4b_branch2a[0][0]              
____________________________________________________________________________________________________
res4b_branch2b (Convolution2D)   (None, 14, 14, 256)   590080      activation_26[0][0]              
____________________________________________________________________________________________________
bn4b_branch2b (BatchNormalizatio (None, 14, 14, 256)   1024        res4b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_27 (Activation)       (None, 14, 14, 256)   0           bn4b_branch2b[0][0]              
____________________________________________________________________________________________________
res4b_branch2c (Convolution2D)   (None, 14, 14, 1024)  263168      activation_27[0][0]              
____________________________________________________________________________________________________
bn4b_branch2c (BatchNormalizatio (None, 14, 14, 1024)  4096        res4b_branch2c[0][0]             
____________________________________________________________________________________________________
merge_9 (Merge)                  (None, 14, 14, 1024)  0           bn4b_branch2c[0][0]              
                                                                   activation_25[0][0]              
____________________________________________________________________________________________________
activation_28 (Activation)       (None, 14, 14, 1024)  0           merge_9[0][0]                    
____________________________________________________________________________________________________
res4c_branch2a (Convolution2D)   (None, 14, 14, 256)   262400      activation_28[0][0]              
____________________________________________________________________________________________________
bn4c_branch2a (BatchNormalizatio (None, 14, 14, 256)   1024        res4c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_29 (Activation)       (None, 14, 14, 256)   0           bn4c_branch2a[0][0]              
____________________________________________________________________________________________________
res4c_branch2b (Convolution2D)   (None, 14, 14, 256)   590080      activation_29[0][0]              
____________________________________________________________________________________________________
bn4c_branch2b (BatchNormalizatio (None, 14, 14, 256)   1024        res4c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_30 (Activation)       (None, 14, 14, 256)   0           bn4c_branch2b[0][0]              
____________________________________________________________________________________________________
res4c_branch2c (Convolution2D)   (None, 14, 14, 1024)  263168      activation_30[0][0]              
____________________________________________________________________________________________________
bn4c_branch2c (BatchNormalizatio (None, 14, 14, 1024)  4096        res4c_branch2c[0][0]             
____________________________________________________________________________________________________
merge_10 (Merge)                 (None, 14, 14, 1024)  0           bn4c_branch2c[0][0]              
                                                                   activation_28[0][0]              
____________________________________________________________________________________________________
activation_31 (Activation)       (None, 14, 14, 1024)  0           merge_10[0][0]                   
____________________________________________________________________________________________________
res4d_branch2a (Convolution2D)   (None, 14, 14, 256)   262400      activation_31[0][0]              
____________________________________________________________________________________________________
bn4d_branch2a (BatchNormalizatio (None, 14, 14, 256)   1024        res4d_branch2a[0][0]             
____________________________________________________________________________________________________
activation_32 (Activation)       (None, 14, 14, 256)   0           bn4d_branch2a[0][0]              
____________________________________________________________________________________________________
res4d_branch2b (Convolution2D)   (None, 14, 14, 256)   590080      activation_32[0][0]              
____________________________________________________________________________________________________
bn4d_branch2b (BatchNormalizatio (None, 14, 14, 256)   1024        res4d_branch2b[0][0]             
____________________________________________________________________________________________________
activation_33 (Activation)       (None, 14, 14, 256)   0           bn4d_branch2b[0][0]              
____________________________________________________________________________________________________
res4d_branch2c (Convolution2D)   (None, 14, 14, 1024)  263168      activation_33[0][0]              
____________________________________________________________________________________________________
bn4d_branch2c (BatchNormalizatio (None, 14, 14, 1024)  4096        res4d_branch2c[0][0]             
____________________________________________________________________________________________________
merge_11 (Merge)                 (None, 14, 14, 1024)  0           bn4d_branch2c[0][0]              
                                                                   activation_31[0][0]              
____________________________________________________________________________________________________
activation_34 (Activation)       (None, 14, 14, 1024)  0           merge_11[0][0]                   
____________________________________________________________________________________________________
res4e_branch2a (Convolution2D)   (None, 14, 14, 256)   262400      activation_34[0][0]              
____________________________________________________________________________________________________
bn4e_branch2a (BatchNormalizatio (None, 14, 14, 256)   1024        res4e_branch2a[0][0]             
____________________________________________________________________________________________________
activation_35 (Activation)       (None, 14, 14, 256)   0           bn4e_branch2a[0][0]              
____________________________________________________________________________________________________
res4e_branch2b (Convolution2D)   (None, 14, 14, 256)   590080      activation_35[0][0]              
____________________________________________________________________________________________________
bn4e_branch2b (BatchNormalizatio (None, 14, 14, 256)   1024        res4e_branch2b[0][0]             
____________________________________________________________________________________________________
activation_36 (Activation)       (None, 14, 14, 256)   0           bn4e_branch2b[0][0]              
____________________________________________________________________________________________________
res4e_branch2c (Convolution2D)   (None, 14, 14, 1024)  263168      activation_36[0][0]              
____________________________________________________________________________________________________
bn4e_branch2c (BatchNormalizatio (None, 14, 14, 1024)  4096        res4e_branch2c[0][0]             
____________________________________________________________________________________________________
merge_12 (Merge)                 (None, 14, 14, 1024)  0           bn4e_branch2c[0][0]              
                                                                   activation_34[0][0]              
____________________________________________________________________________________________________
activation_37 (Activation)       (None, 14, 14, 1024)  0           merge_12[0][0]                   
____________________________________________________________________________________________________
res4f_branch2a (Convolution2D)   (None, 14, 14, 256)   262400      activation_37[0][0]              
____________________________________________________________________________________________________
bn4f_branch2a (BatchNormalizatio (None, 14, 14, 256)   1024        res4f_branch2a[0][0]             
____________________________________________________________________________________________________
activation_38 (Activation)       (None, 14, 14, 256)   0           bn4f_branch2a[0][0]              
____________________________________________________________________________________________________
res4f_branch2b (Convolution2D)   (None, 14, 14, 256)   590080      activation_38[0][0]              
____________________________________________________________________________________________________
bn4f_branch2b (BatchNormalizatio (None, 14, 14, 256)   1024        res4f_branch2b[0][0]             
____________________________________________________________________________________________________
activation_39 (Activation)       (None, 14, 14, 256)   0           bn4f_branch2b[0][0]              
____________________________________________________________________________________________________
res4f_branch2c (Convolution2D)   (None, 14, 14, 1024)  263168      activation_39[0][0]              
____________________________________________________________________________________________________
bn4f_branch2c (BatchNormalizatio (None, 14, 14, 1024)  4096        res4f_branch2c[0][0]             
____________________________________________________________________________________________________
merge_13 (Merge)                 (None, 14, 14, 1024)  0           bn4f_branch2c[0][0]              
                                                                   activation_37[0][0]              
____________________________________________________________________________________________________
activation_40 (Activation)       (None, 14, 14, 1024)  0           merge_13[0][0]                   
____________________________________________________________________________________________________
res5a_branch2a (Convolution2D)   (None, 7, 7, 512)     524800      activation_40[0][0]              
____________________________________________________________________________________________________
bn5a_branch2a (BatchNormalizatio (None, 7, 7, 512)     2048        res5a_branch2a[0][0]             
____________________________________________________________________________________________________
activation_41 (Activation)       (None, 7, 7, 512)     0           bn5a_branch2a[0][0]              
____________________________________________________________________________________________________
res5a_branch2b (Convolution2D)   (None, 7, 7, 512)     2359808     activation_41[0][0]              
____________________________________________________________________________________________________
bn5a_branch2b (BatchNormalizatio (None, 7, 7, 512)     2048        res5a_branch2b[0][0]             
____________________________________________________________________________________________________
activation_42 (Activation)       (None, 7, 7, 512)     0           bn5a_branch2b[0][0]              
____________________________________________________________________________________________________
res5a_branch2c (Convolution2D)   (None, 7, 7, 2048)    1050624     activation_42[0][0]              
____________________________________________________________________________________________________
res5a_branch1 (Convolution2D)    (None, 7, 7, 2048)    2099200     activation_40[0][0]              
____________________________________________________________________________________________________
bn5a_branch2c (BatchNormalizatio (None, 7, 7, 2048)    8192        res5a_branch2c[0][0]             
____________________________________________________________________________________________________
bn5a_branch1 (BatchNormalization (None, 7, 7, 2048)    8192        res5a_branch1[0][0]              
____________________________________________________________________________________________________
merge_14 (Merge)                 (None, 7, 7, 2048)    0           bn5a_branch2c[0][0]              
                                                                   bn5a_branch1[0][0]               
____________________________________________________________________________________________________
activation_43 (Activation)       (None, 7, 7, 2048)    0           merge_14[0][0]                   
____________________________________________________________________________________________________
res5b_branch2a (Convolution2D)   (None, 7, 7, 512)     1049088     activation_43[0][0]              
____________________________________________________________________________________________________
bn5b_branch2a (BatchNormalizatio (None, 7, 7, 512)     2048        res5b_branch2a[0][0]             
____________________________________________________________________________________________________
activation_44 (Activation)       (None, 7, 7, 512)     0           bn5b_branch2a[0][0]              
____________________________________________________________________________________________________
res5b_branch2b (Convolution2D)   (None, 7, 7, 512)     2359808     activation_44[0][0]              
____________________________________________________________________________________________________
bn5b_branch2b (BatchNormalizatio (None, 7, 7, 512)     2048        res5b_branch2b[0][0]             
____________________________________________________________________________________________________
activation_45 (Activation)       (None, 7, 7, 512)     0           bn5b_branch2b[0][0]              
____________________________________________________________________________________________________
res5b_branch2c (Convolution2D)   (None, 7, 7, 2048)    1050624     activation_45[0][0]              
____________________________________________________________________________________________________
bn5b_branch2c (BatchNormalizatio (None, 7, 7, 2048)    8192        res5b_branch2c[0][0]             
____________________________________________________________________________________________________
merge_15 (Merge)                 (None, 7, 7, 2048)    0           bn5b_branch2c[0][0]              
                                                                   activation_43[0][0]              
____________________________________________________________________________________________________
activation_46 (Activation)       (None, 7, 7, 2048)    0           merge_15[0][0]                   
____________________________________________________________________________________________________
res5c_branch2a (Convolution2D)   (None, 7, 7, 512)     1049088     activation_46[0][0]              
____________________________________________________________________________________________________
bn5c_branch2a (BatchNormalizatio (None, 7, 7, 512)     2048        res5c_branch2a[0][0]             
____________________________________________________________________________________________________
activation_47 (Activation)       (None, 7, 7, 512)     0           bn5c_branch2a[0][0]              
____________________________________________________________________________________________________
res5c_branch2b (Convolution2D)   (None, 7, 7, 512)     2359808     activation_47[0][0]              
____________________________________________________________________________________________________
bn5c_branch2b (BatchNormalizatio (None, 7, 7, 512)     2048        res5c_branch2b[0][0]             
____________________________________________________________________________________________________
activation_48 (Activation)       (None, 7, 7, 512)     0           bn5c_branch2b[0][0]              
____________________________________________________________________________________________________
res5c_branch2c (Convolution2D)   (None, 7, 7, 2048)    1050624     activation_48[0][0]              
____________________________________________________________________________________________________
bn5c_branch2c (BatchNormalizatio (None, 7, 7, 2048)    8192        res5c_branch2c[0][0]             
____________________________________________________________________________________________________
merge_16 (Merge)                 (None, 7, 7, 2048)    0           bn5c_branch2c[0][0]              
                                                                   activation_46[0][0]              
____________________________________________________________________________________________________
activation_49 (Activation)       (None, 7, 7, 2048)    0           merge_16[0][0]                   
____________________________________________________________________________________________________
avg_pool (AveragePooling2D)      (None, 1, 1, 2048)    0           activation_49[0][0]              
====================================================================================================
Total params: 23,587,712
Trainable params: 23,534,592
Non-trainable params: 53,120
____________________________________________________________________________________________________
```