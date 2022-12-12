clear all; close all; clc;
% 
%%%%%%%%%% plot  frequency Fig.1 %%%%%%%%%%%%%

for n=0:6
% num=xlsread('nadyr_ddpg_sim_0.csv');
% eval(["num=xlsread('nadyr_ddpg_sim_",int2str(n),".csv');"])
num=xlsread(sprintf('nadyr_ddpg_sim_%d.csv',n));
% eval(['save myfile',int2str(n),'.mat'])
% num=xlsread('nadyr_ddpg_sim_0');
figure(1)
p=plot(num(:,1),'LineWidth', 1');
hold on
legend('ls=0','ls=100','ls=200','ls=300','ls=400','ls=500','ls=600')
xlabel('Time (seconds)')
ylabel('Frequency (Hz)')
set(gca,'FontSize',12);
grid on;
xlim([2.5 100])
ylim([59.35 59.8])
end

for n=0:6
% num=xlsread('nadyr_ddpg_sim_0.csv');
% eval(["num=xlsread('nadyr_ddpg_sim_",int2str(n),".csv');"])
num=xlsread(sprintf('nadyr_td3_sim_%d.csv',n));
% eval(['save myfile',int2str(n),'.mat'])
% num=xlsread('nadyr_ddpg_sim_0');
figure(2)
p=plot(num(:,1),'LineWidth', 1');
hold on
legend('ls=0','ls=100','ls=200','ls=300','ls=400','ls=500','ls=600')
xlabel('Time (seconds)')
ylabel('Frequency (Hz)')
set(gca,'FontSize',12);
grid on;
xlim([2.5 100])
ylim([59.35 59.8])
end