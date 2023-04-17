clc;
clear;
load('D:\�ű�\MatrixSketch2\GeantStore.mat');
[allDataI1,allDataI2,allDataI3]=size(GeantDataNorm2);%ȫ�����ݵ�ά��
%%
%���������� 
numSliceForSketchStorage=25;%sketch�洢���ٸ�slice
trainingSliceNum=500; %ǰ�ڵ�ѵ���õ��쳣������Slice  1500   672  500
deteionSliceNum=allDataI3-trainingSliceNum; %���Slice����Ŀ��������allDataI3-trainingSliceNum
anomalySliceNum=round(deteionSliceNum*0.5); %���Slice���쳣����Ŀ  round(deteionSliceNum*0.4)

mu=0.0001;%ƽ��ֵ0.00004  0.0001  0.00000000000000004-0.000000000000004
sigma=0.0001;%����0.00009  0.0001  0.000000000000009
abnormalPointNum =round(allDataI1*allDataI2*0.1);%ÿ����ע���쳣�ĸ�������53

anomalySlicePos=randperm(deteionSliceNum,anomalySliceNum)+trainingSliceNum;%randperm(m,n)�������ã���1-m���������n�����ظ�����
anomalySlicePos=sort(anomalySlicePos);%����ʹ�����ѡ����쳣����Ŵ�С��������
rightIsAbnormal=0;%��ȷ���쳣 �ж�Ϊ�쳣
wrongIsAbnormal=0;%�������� �ж�Ϊ�쳣

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

%%˼��һ��SVD�ֽ��U��PCA�ֽ�Ĺ�ϵ
%%
for i=1:trainingSliceNum
    slice=GeantDataNorm2(:,:,i);%ĳһ��ľ���
    mode1Matrix=[mode1Matrix,slice];
    mode2Matrix=[mode2Matrix,slice'];
    %slice������
    tempV=reshape(slice,1,allDataI1*allDataI2);
    mode3Matrix=[mode3Matrix;tempV];
    if(i>numSliceForSketchStorage)
        [curScore1]=deteAnomaly(1,slice,allDataI1,allDataI2,mode1Matrix,topK1);
        [curScore2]=deteAnomaly(2,slice,allDataI1,allDataI2,mode2Matrix,topK2);
        [curScore3]=deteAnomaly(3,slice,allDataI1,allDataI2,mode3Matrix,topK3);
        
        trainingScore1(1,trainingCount)=curScore1;%���Ե���չ�����ʱ��ע���޸�
        trainingScore2(1,trainingCount)=curScore2;%���Ե���չ�����ʱ��ע���޸�
        trainingScore3(1,trainingCount)=curScore3;%���Ե���չ�����ʱ��ע���޸�
        trainingCount=trainingCount+1;
 
    end
end

trainingScore1=sort(trainingScore1);
trainingScore2=sort(trainingScore2);
trainingScore3=sort(trainingScore3);
flagScore1=trainingScore1(1,trainingSliceNum-numSliceForSketchStorage);
flagScore2=trainingScore2(1,trainingSliceNum-numSliceForSketchStorage);
flagScore3=trainingScore3(1,trainingSliceNum-numSliceForSketchStorage);


%���쳣���
disp("deteionSliceNum="+deteionSliceNum);
for i=1:deteionSliceNum
    disp("trainingSliceNum+i="+(trainingSliceNum+i));
    slice=GeantDataNorm2(:,:,trainingSliceNum+i);%ĳһ��ľ���
     if(ismember(trainingSliceNum+i,anomalySlicePos)==1)%�����������쳣��
        %�����쳣ֵ
        outliers=normrnd(mu,sigma,[1,abnormalPointNum]);%�Ӿ�ֵ����Ϊ mu �ͱ�׼�����Ϊ sigma ����̬�ֲ������� 1��outliersNum���쳣ֵ
        pos=randperm(allDataI1*allDataI2, abnormalPointNum);%randperm(m,n)�������ã���1-m���������n�����ظ�����
        pos=sort(pos);%����ʹ�����ѡ����쳣����Ŵ�С��������
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
                ME = MException(errID,msgtext);%ֱ������һ��
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
        if(curScore2>flagScore2)%��ǰ���������ж�Ϊ���쳣
           rightIsAbnormal=rightIsAbnormal+1;
        else
            mode1Matrix=[mode1Matrix,slice];
            mode2Matrix=[mode2Matrix,slice'];
            %slice������
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
         if(curScore2>flagScore2)%�ж�Ϊ �쳣
           %�����������ж�Ϊ�쳣       
           wrongIsAbnormal=wrongIsAbnormal+1; 
         else
           mode1Matrix=[mode1Matrix,slice];
           mode2Matrix=[mode2Matrix,slice'];
            %slice������
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


disp("�쳣�ж���ȷ�ʣ�Խ��Խ�ã�"+(rightIsAbnormal/anomalySliceNum));
disp("�쳣�жϴ����ʣ�Խ��Խ�ã�"+(wrongIsAbnormal/(deteionSliceNum-anomalySliceNum)));
