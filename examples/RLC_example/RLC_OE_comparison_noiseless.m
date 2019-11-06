clear;
clc;

%% Read data table 

data_path = fullfile("data", "RLC_data_id.csv");
data_table = readtable(data_path);


u =  data_table.V_IN;
y = data_table.V_C;
t = data_table.time;
Ts = t(2) - t(1);

%% Add noise %%
STD_V = 0.0;
y_meas = y + randn(size(y))*STD_V;

%% Identification data %%

data_id = iddata(y_meas,u,Ts);
model_oe = oe(data_id, 'nb',2, 'nf', 2);
y_sim = sim(model_oe, data_id);
y_sim_val = y_sim.OutputData;

loss = mean((y - y_sim_val).^2);

%% Plot data %%

plot(t, y);
hold on;
plot(t, y_sim_val);
legend('Measured', 'Simulated');


%%
SSE = sum((y - y_sim_val).^2);
y_mean = mean(y);
SST = sum((y - y_mean).^2);

R_sq = 1 - SSE/SST;
