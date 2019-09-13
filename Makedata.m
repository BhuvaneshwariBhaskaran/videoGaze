
clc;
clear all;
%%
data=load('train_super_annotations.mat');

train_path=data.train_path(1:500);
train_gaze=data.train_gaze(1:500);
train_bbox=data.train_bbox(1:500);
train_meta=data.train_meta(1:500);
train_eyes=data.train_eyes(1:500);

%%
%train_subset=[subset_path,subset_gaze, subset_bbox,subset_meta,subset_eyes]

save('train_annotations.mat','train_eyes','train_gaze','train_bbox','train_meta','train_path')
%%

