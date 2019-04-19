%% Read data
%load('D:\E\superalarm_data\TrainTestData_allFea.mat')
load('adaboost_auc.mat')
[para1,para2] = size(CA_TrainData_allFea);
%% main 
MaxNumSplits = [1,2,3,4,5];
tree_num = [20,50,100,150,200];
fold_num = 5;
B = cell(7,45,5);
result = [];
best_auc = 0;
for i = 1:paral
    for j = 1:para2
          for n = 1:length(MaxNumSplits)
              for m = 1:length(tree_num)
                a = 0;
                for fold = 1:5
                   a = a + A(i,j,n,m,fold);
                end
                result = [result;a/5];
                if (a/5) > best_auc
                    best_auc = a/5;
                    index = [i,j,n,m];
                end
            end
        end
    end
end
lable1 = CA_TrainData_allFea{index(1),index(2)};
lable0 = Control_TrainData_allFea{index(1),index(1)};

% cat the lable0 and lable1
train_data = cat(1,lable0,lable1);
X_train = train_data(:,1:1:281);
y_train = train_data(:,282);

test_lable1 = CA_TestData_allFea{index(1),index(2)};
test_lable0 = Control_TestData_allFea{index(1),index(1)};
test_data = cat(1,test_lable0,test_lable1);
X_test = test_data(:,1:1:281);
y_test = test_data(:,282);
% AdaBoost train
ens = fitcensemble(X_train, y_train,'Method','AdaBoostM1','NumLearningCycles',tree_num(index(3)),'Learners',templateTree('MaxNumSplits',MaxNumSplits(index(4))),'ScoreTransform','logit');

[predict_label,score] = predict(ens, X_test);
AUC = calculate_auc(score(:,2),y_test);
AUC
EVAL = Evaluate(predict_label,y_test);
EVAL