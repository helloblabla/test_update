clc;
clear;
load('D:\脚本\MatrixSketch2\GeantStore.mat');
[allDataI1,allDataI2,allDataI3]=size(GeantDataNorm2);%全部数据的维度
%%
%参数设置区 
numSliceForSketchStorage=25;%sketch存储多少个slice
trainingSliceNum=500; %前期的训练得到异常分数的Slice  1500   672  500
deteionSliceNum=allDataI3-trainingSliceNum; %检测Slice的数目，不超过allDataI3-trainingSliceNum
anomalySliceNum=round(deteionSliceNum*0.5); %检测Slice中异常的数目  round(deteionSliceNum*0.4)

mu=0.0001;%平均值0.00004  0.0001  0.00000000000000004-0.000000000000004
sigma=0.0001;%方差0.00009  0.0001  0.000000000000009
abnormalPointNum =round(allDataI1*allDataI2*0.1);%每个面注入异常的个数，即53

anomalySlicePos=randperm(deteionSliceNum,anomalySliceNum)+trainingSliceNum;%randperm(m,n)函数作用：从1-m中随机产生n个不重复的数
anomalySlicePos=sort(anomalySlicePos);%排序，使得随机选择的异常面序号从小到大排序
rightIsAbnormal=0;%正确将异常 判断为异常
wrongIsAbnormal=0;%错误将正常 判断为异常

mode1Matrix=zeros(0,0);
mode2Matrix=zeros(0,0);
mode3Matrix=zeros(0,0);

curScore1=0;
curScore2=0;
curScore3=0;

topK1=1;
topK2=1;
topK3=1;

trainingScore1=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingScore2=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingScore3=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingScoreIndex=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingCount=1;

abnorScoreArr1=zeros(1,anomalySliceNum);
abnorScoreArr2=zeros(1,anomalySliceNum);
abnorScoreArr3=zeros(1,anomalySliceNum);
abnorScoreArr_Totally=zeros(1,anomalySliceNum);
abnorCount=1;
norScoreArr1=zeros(1,deteionSliceNum-anomalySliceNum); 
norScoreArr2=zeros(1,deteionSliceNum-anomalySliceNum);
norScoreArr3=zeros(1,deteionSliceNum-anomalySliceNum); 
norScoreArr_Totally=zeros(1,deteionSliceNum-anomalySliceNum); 
norCount=1;

%%思考一下SVD分解的U和PCA分解的关系
%%
for i=1:trainingSliceNum
    slice=GeantDataNorm2(:,:,i);%某一面的矩阵
    mode1Matrix=[mode1Matrix,slice];
    mode2Matrix=[mode2Matrix,slice'];
    %slice向量化
    tempV=reshape(slice,1,allDataI1*allDataI2);
    mode3Matrix=[mode3Matrix;tempV];
    if(i>numSliceForSketchStorage)
        [curScore1]=deteAnomaly(1,slice,allDataI1,allDataI2,mode1Matrix,topK1);
        [curScore2]=deteAnomaly(2,slice,allDataI1,allDataI2,mode2Matrix,topK2);
        [curScore3]=deteAnomaly(3,slice,allDataI1,allDataI2,mode3Matrix,topK3);
        
        trainingScore1(1,trainingCount)=curScore1;%测试单个展开面的时候注意修改
        trainingScore2(1,trainingCount)=curScore2;%测试单个展开面的时候注意修改
        trainingScore3(1,trainingCount)=curScore3;%测试单个展开面的时候注意修改
        trainingCount=trainingCount+1;
 
    end
end

trainingScore1=sort(trainingScore1);
trainingScore2=sort(trainingScore2);
trainingScore3=sort(trainingScore3);
flagScore1=trainingScore1(1,trainingSliceNum-numSliceForSketchStorage);
flagScore2=trainingScore2(1,trainingSliceNum-numSliceForSketchStorage);
flagScore3=trainingScore3(1,trainingSliceNum-numSliceForSketchStorage);


%做异常检测
disp("deteionSliceNum="+deteionSliceNum);
for i=1:deteionSliceNum
    disp("trainingSliceNum+i="+(trainingSliceNum+i));
    slice=GeantDataNorm2(:,:,trainingSliceNum+i);%某一面的矩阵
     if(ismember(trainingSliceNum+i,anomalySlicePos)==1)%如果这个面是异常面
        %生成异常值
        outliers=normrnd(mu,sigma,[1,abnormalPointNum]);%从均值参数为 mu 和标准差参数为 sigma 的正态分布中生成 1×outliersNum的异常值
        pos=randperm(allDataI1*allDataI2, abnormalPointNum);%randperm(m,n)函数作用：从1-m中随机产生n个不重复的数
        pos=sort(pos);%排序，使得随机选择的异常面序号从小到大排序
        for j=1:abnormalPointNum
            posX1=floor(pos(1,j)./allDataI1)+1;
            posX2=mod(pos(1,j),allDataI2);
            if(posX2==0)
                posX1=posX1-1;
                posX2=allDataI2;
            end 
           if(posX1<1 || posX1>23 ||  posX2<1 || posX2>23)         
                errID = 'myComponent:inputError';
                msgtext = 'Input does not have the expected format, please check the parameter above.';
                ME = MException(errID,msgtext);%直接生成一个
               throw(ME);
           end 
           slice(posX1,posX2)=slice(posX1,posX2)+outliers(1,j); 
        end
        [curScore1]=deteAnomaly(1,slice,allDataI1,allDataI2,mode1Matrix,topK1);
        [curScore2]=deteAnomaly(2,slice,allDataI1,allDataI2,mode2Matrix,topK2);
        [curScore3]=deteAnomaly(3,slice,allDataI1,allDataI2,mode3Matrix,topK3);       
        abnorScoreArr1(1,abnorCount)=curScore1;
        abnorScoreArr2(1,abnorCount)=curScore2;
        abnorScoreArr3(1,abnorCount)=curScore3;
        abnorCount=abnorCount+1;
        if(curScore2>flagScore2)%当前分数过大，判断为：异常
           rightIsAbnormal=rightIsAbnormal+1;
        else
            mode1Matrix=[mode1Matrix,slice];
            mode2Matrix=[mode2Matrix,slice'];
            %slice向量化
            tempV=reshape(slice,1,allDataI1*allDataI2);
            mode3Matrix=[mode3Matrix;tempV];           
        end
     else
          [curScore1]=deteAnomaly(1,slice,allDataI1,allDataI2,mode1Matrix,topK1);
          [curScore2]=deteAnomaly(2,slice,allDataI1,allDataI2,mode2Matrix,topK2);
          [curScore3]=deteAnomaly(3,slice,allDataI1,allDataI2,mode3Matrix,topK3); 
          norScoreArr1(1,norCount)=curScore1;
          norScoreArr2(1,norCount)=curScore2;
          norScoreArr3(1,norCount)=curScore3;
          norCount=norCount+1;
         if(curScore2>flagScore2)%判断为 异常
           %正常，但是判断为异常       
           wrongIsAbnormal=wrongIsAbnormal+1; 
         else
           mode1Matrix=[mode1Matrix,slice];
           mode2Matrix=[mode2Matrix,slice'];
            %slice向量化
           tempV=reshape(slice,allDataI1*allDataI2,1);
           mode3Matrix=[mode3Matrix;tempV]; 
        end 
     end
end

abnorScoreArr1=sort(abnorScoreArr1);
abnorScoreArr2=sort(abnorScoreArr2);
abnorScoreArr3=sort(abnorScoreArr3);

norScoreArr1=sort(norScoreArr1);
norScoreArr2=sort(norScoreArr2);
norScoreArr3=sort(norScoreArr3);


disp("异常判断正确率（越高越好）"+(rightIsAbnormal/anomalySliceNum));
disp("异常判断错误率（越低越好）"+(wrongIsAbnormal/(deteionSliceNum-anomalySliceNum)));
