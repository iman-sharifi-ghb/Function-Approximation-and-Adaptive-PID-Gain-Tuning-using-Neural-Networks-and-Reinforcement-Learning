clc;
clear;
close all

global beta0 T
global sigma Amp
beta0 = 0;
T     = 0.01;
sigma = 1;
Amp   = -1;

dr = 0.9;    % discount rate
LR = 0.1;     % Learning Rate
x0 = [1;1];   % Initial Condition
Tf = 200;
t  = 0:T:Tf;
N = Tf/T;

% Initialization
% u  = zeros(1, N+1);
u  = sin(2*pi/100*t);
e1 = zeros(1, N+1);
e2 = zeros(1, N+1);

ym = zeros(N+1, 1);
V  = ones(N+1, 1);
x1 = x0(1)*ones(1, N+1);
x2 = x0(2)*ones(1, N+1);
y  = x2;
Outputs = x0;

% Neural Network structure
n  = 3;      % Number of NN box Inputs for function approximation
N1 = 10;     % Number of nourons

% Initialize Weights
W1_Critic = zeros(n, N1);
W1_Actor  = zeros(n, N1);
W2_Critic = zeros(N1+1,1);
W2_Actor  = zeros(N1+1,1);

% Main Loop

for k = 1:N
    
    Outputs = NonLinDynamic(Outputs, u(k));
    Outputs = Outputs + 0.01*norm(Outputs)*randn(size(Outputs));
    
    x1(k+1) = Outputs(1);
    x2(k+1) = Outputs(2);
    
    y(k+1)  = Outputs(2);
    
    % Forward net
    history_vector    = [x1(k), x2(k), u(k)].';
    [V(k+1), ym(k+1), Phi_Critic, Phi_Actor] = RL_Agent(history_vector, W1_Critic, W2_Critic, W1_Actor, W2_Actor);
    
    Sigma_V = 0.01*sigmoid(-2*V(k));
    ym(k+1) = ym(k+1) + Sigma_V*randn();
    % Get reward
    r = Reward(y(k+1), ym(k+1));
    
    % TD error
    delta_TD = r + dr* V(k+1) - V(k);
    e        = y(k+1)-ym(k+1);
    % Error  = 1/2*(delta_TD)^2
    
    % Backward Updates for Critic
    W2_Critic = W2_Critic - LR*(delta_TD)*(-1)*Phi_Critic;
    for i=1:n
        for j=1:N1
            W1_Critic(i,j) = W1_Critic(i,j) - LR*(delta_TD)*(-1)*W2_Critic(j)*Phi_Critic(j)*(1-Phi_Critic(j))*history_vector(i);
        end
    end
    % Backward Updates for Actor
    W2_Actor = W2_Actor - LR*((delta_TD)*(2*e/sigma^2*r)-e)*Phi_Actor;
    for i=1:n
        for j=1:N1
            W1_Actor(i,j) = W1_Actor(i,j) - LR*((delta_TD)*(2*e/sigma^2*r)-e)*W2_Actor(j)*Phi_Actor(j)*(1-Phi_Actor(j))*history_vector(i);
        end
    end
    Amp = Amp - LR*delta_TD*exp(-e^2/sigma^2);
    
end
%% plot Outpots
figure;
plot(t, y, 'LineWidth', 2)
hold on, grid on
plot(t, ym, 'LineWidth', 2)
xlabel('Time'), ylabel('Outputs')
legend('y', 'ym')
% axis([0 10 -5 5])

%% Functions
function [V, ym, Ho_Critic, Ho_Actor] = RL_Agent(X, W1_Critic, W2_Critic, W1_Actor, W2_Actor)

    % Critic Part
    hi_Critic = (X'*W1_Critic)';
    ho_Critic = sigmoid(hi_Critic);
    Ho_Critic = [ho_Critic;1];
    V         = W2_Critic'*Ho_Critic;

    % Actor Part
    hi_Actor = (X'*W1_Actor)';
    ho_Actor = sigmoid(hi_Actor);
    Ho_Actor = [ho_Actor;1];
    ym       = W2_Actor'*Ho_Actor;

end
function r = Reward(y, ym)
    global sigma Amp
    r = Amp*exp(-(y-ym)^2/sigma^2);
end
function y = sigmoid(z)
    y = 1./(1+exp(-z));
end

function Out = NonLinDynamic(x,u)
    
    global T
    Da = 0.72;
    B = 8;
    gama = 20;
    beta = 0.3;
    
    x1 = x(1);
    x2 = x(2);

    x1_prime = x1+T*(-x1 + Da*(1-x1)*exp(x2/(1+x2/gama)));
    
    x2_prime = x2+T*(-x2 + B*Da*(1-x1)*exp(x2/(1+x2/gama))) + beta*(u-x2);

    Out        = [x1_prime;x2_prime];
    
end

