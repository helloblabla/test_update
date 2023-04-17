function [curScore]=deteAnomaly(mode,slice,allDataI1,allDataI2,modeMatrix,topK)
[mode1Matrix_I1,mode1Matrix_I2]=size(modeMatrix);
if(topK > min(mode1Matrix_I1,mode1Matrix_I2))
    errID = 'myComponent:printError';
    msgtext = 'output does not have the expected format.';
    ME = MException(errID,msgtext);%直接生成一个
    throw(ME);
end
if(mode==1 || mode==2)
    if(mode==2)
        slice=slice';
    end
    %异常检测:
    [U,~,~]=svds(modeMatrix,topK);
    expression=slice-U*U'*slice;
    curScore=norm(expression,'fro');    
else
    [~,~,V]=svds(modeMatrix,topK);
    vecSlice=reshape(slice,allDataI1*allDataI2,1);
    expression=(vecSlice-V*V'*vecSlice);
    curScore=norm(expression,2);   
end
end