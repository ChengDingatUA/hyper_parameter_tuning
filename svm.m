%% Read data
CA_TrainData_allFea = load('TrainTestData_allFea.mat', 'CA_TrainData_allFea');
Control_TrainData_allFea = load('TrainTestData_allFea.mat', 'Control_TrainData_allFea');
%% main 
A = zeros(7,45,5,4,4);
C = [1,2,3,4,5];
gammma = [0.125,0.25,0.375,0.5];
for i = 1:7
    for j = 1:45
          for n = 1:size(C,2)
              for m = 1:4
                for k = 1:5
                   % lable1 - CA  lable0 -- Control   
                   lable1 = CA_TrainData_allFea.CA_TrainData_allFea{i,j};
                   lable0 = Control_TrainData_allFea.Control_TrainData_allFea{i,j};
                   % get the size of lable1 and lable0
                   lable1_size = size(CA_TrainData_allFea.CA_TrainData_allFea{i,j});
                   lable0_size = size(Control_TrainData_allFea.Control_TrainData_allFea{i,j});
                   % cat the lable0 and lable1
                   train_all = cat(1,lable0,lable1);
                   [X_train, y_train,  X_valid, y_valid] = split_train_valid(train_all);
                   %[X_train, y_train,  X_valid, y_valid] = resplit_data(train_all);
                   
                   % SVM train
                   model= svmtrain(y_train,X_train,strcat('-q -s 0 -t 2 -b 1 -c ',32,num2str(C(n)),32,'-g',32,num2str(gammma(m))));
                   % SVM predict
                   [prelabel,accuracy,decision_values]=svmpredict(y_valid, X_valid,model,'-q -b 1');
                   result = calculate_auc(decision_values(:,2),y_valid);
                   A(i,j,k,n,m) = result;
                   B{i,j,k} = cat(2,prelabel,y_valid);
                end
            end
        end
    end
end
save('svm_auc.mat','A'); 
save('svm_detail.mat','B'); 


