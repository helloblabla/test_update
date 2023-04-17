clc;
clear;


disp("异常判断正确率（越高越好）"+(rightIsAbnormal/anomalySliceNum));
disp("异常判断错误率（越低越好）"+(wrongIsAbnormal/(deteionSliceNum-anomalySliceNum)));
