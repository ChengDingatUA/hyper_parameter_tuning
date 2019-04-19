%% Read data
load('D:\E\superalarm_data\TrainTestData_allFea.mat')
[para1,para2] = size(CA_TrainData_allFea);
%% main 
MaxNumSplits = [1,2,3,4,5];
tree_num = [20,50,100,150,200];
fold_num = 5;
A = zeros(para1,para2,length(MaxNumSplits),length(tree_num),fold_num);
B = cell(7,45,fold_num);
for i = 1:para1
    for j = 1:para2
          for n = 1:length(MaxNumSplits)
              for m = 1:length(tree_num)
                parfor fold = 1:5
                   % lable1 - CA  lable0 -- Control   
                   lable1 = CA_TrainData_allFea{i,j};
                   lable0 = Control_TrainData_allFea{i,j};
                   % cat the lable0 and lable1
                   train_all = cat(1,lable0,lable1);
                   [X_train, y_train,  X_valid, y_valid] = resplit_data(train_all);
                   ens = fitcensemble(X_train, y_train,'Method','AdaBoostM1','NumLearningCycles',tree_num(m),'Learners',templateTree('MaxNumSplits',MaxNumSplits(n)),'ScoreTransform','logit');
                     % AdaBoost train
                   [predict_label,score] = predict(ens, X_valid);
                   AUC = calculate_auc(score(:,2),y_valid);
                   AUC
                   EVAL = Evaluate(predict_label,y_valid);
                   EVAL
                   A(i,j,n,m,fold) = AUC;
                   B{i,j,fold} = cat(2,score(:,2),y_valid);
                end
            end
        end
    end
end
save('adaboost_auc.mat','A'); 
save('adaboost_output.mat','B'); 