clear;
clc;

%% Read data table 

data_path = fullfile("data", "RLC_data_sat_FE.csv");
data_table = readtable(data_path);


u =  data_table.V_IN;
y = data_table.V_C;
t = data_table.time;
Ts = t(2) - t(1);

%% Identification data %%

data_id = iddata(y,u,Ts);
model_oe = oe(data_id, 'nb',2, 'nf', 2);
y_sim = sim(model_oe, data_id);

loss = mean(y )
