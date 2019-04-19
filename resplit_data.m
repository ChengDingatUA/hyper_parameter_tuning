function [final_train,final_train_lable,final_val,final_val_lable] = resplit_data(train_data)

EID = unique(train_data(:,end));
len_eid = length(EID);

hierarchy_data = cell(1,len_eid);
for pat = 1:len_eid
   temp_data = train_data(train_data(:,end)== EID(pat),:);
   hierarchy_data{1,pat} = temp_data;
end

%save('hierarchy_data.mat','hierarchy_data');
n = length(hierarchy_data);
rate = int16(0.8*n);
index = 1:1:n;
train_index = index(randperm(n,rate));
validation_index = setdiff(index,train_index);
train = hierarchy_data(:,train_index);
validation = hierarchy_data(:,validation_index);
n = length(hierarchy_data);
num_features = 282;
rate = int16(0.8*n);
index = 1:1:n;
train_index = index(randperm(n,rate));
validation_index = setdiff(index,train_index);
train = hierarchy_data(:,train_index);
validation = hierarchy_data(:,validation_index);

final_train = [];
final_val = [];
num_train = length(train);
num_validation = length(validation);

for ii = 1:num_train
    curr_pat = train(:,ii);
    curr_pat = curr_pat{1,1};
    final_train = [final_train;curr_pat(:,1:1:num_features)];
end

for ii = 1:num_validation
    curr_pat = validation(:,ii);
    curr_pat = curr_pat{1,1};
    final_val = [final_val;curr_pat(:,1:1:num_features)];
end
final_val_lable = final_val(:,282);
final_train_lable = final_train(:,282);
final_val = final_val(:,1:281);
final_train = final_train(:,1:281);



