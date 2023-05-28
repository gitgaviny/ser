load F:\singlecorpus\data/imse_1.mat
load F:\singlecorpus\data/imse_2.mat
load F:\singlecorpus\data/imse_3.mat
load F:\singlecorpus\data/imse_4.mat
load F:\singlecorpus\data/imse_5.mat

train2345_data=[A2;A3;A4;A5];
train2345_label=[y2;y3;y4;y5];
train2345_gender=[g2;g3;g4;g5];
train2345_len=[lenS2;lenS3;lenS4;lenS5];
test1_data=[A1];
test1_label=[y1];
test1_gender=[g1];
test1_len=[lenS1];

train1345_data=[A1;A3;A4;A5];
train1345_label=[y1;y3;y4;y5];
train1345_gender=[g1;g3;g4;g5];
train1345_len=[lenS1;lenS3;lenS4;lenS5];
test2_data=[A2];
test2_label=[y2];
test2_gender=[g2];
test2_len=[lenS2];

train1245_data=[A1;A2;A4;A5];
train1245_label=[y1;y2;y4;y5];
train1245_gender=[g1;g2;g4;g5];
train1245_len=[lenS1;lenS2;lenS4;lenS5];
test3_data=[A3];
test3_label=[y3];
test3_gender=[g3];
test3_len=[lenS3];

train1235_data=[A1;A2;A3;A5];
train1235_label=[y1;y2;y3;y5];
train1235_gender=[g1;g2;g3;g5];
train1235_len=[lenS1;lenS2;lenS3;lenS5];
test4_data=[A4];
test4_label=[y4];
test4_gender=[g4];
test4_len=[lenS4];

train1234_data=[A1;A2;A3;A4];
train1234_label=[y1;y2;y3;y4];
train1234_gender=[g1;g2;g3;g4];
train1234_len=[lenS1;lenS2;lenS3;lenS4];
test5_data=[A5];
test5_label=[y5];
test5_gender=[g5];
test5_len=[lenS5];

                          
save F:\singlecorpus\data\maked/train2345.mat train2345_data train2345_label train2345_gender train2345_len -v7.3
save F:\singlecorpus\data\maked/test1.mat test1_data test1_label test1_gender test1_len -v7.3
save F:\singlecorpus\data\maked/train1345.mat train1345_data train1345_label train1345_gender train1345_len -v7.3
save F:\singlecorpus\data\maked/test2.mat test2_data test2_label test2_gender test2_len -v7.3
save F:\singlecorpus\data\maked/train1245.mat train1245_data train1245_label train1245_gender train1245_len -v7.3
save F:\singlecorpus\data\maked/test3.mat test3_data test3_label test3_gender test3_len -v7.3
save F:\singlecorpus\data\maked/train1235.mat train1235_data train1235_label train1235_gender train1235_len -v7.3
save F:\singlecorpus\data\maked/test4.mat test4_data test4_label test4_gender test4_len -v7.3
save F:\singlecorpus\data\maked/train1234.mat train1234_data train1234_label train1234_gender train1234_len -v7.3
save F:\singlecorpus\data\maked/test5.mat test5_data test5_label test5_gender test5_len -v7.3