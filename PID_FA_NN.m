clc;
clear;
close all

global beta0 T
beta0 = 0;
T     = 0.01;

LR = 0.1;       % Learning Rate
x0 = [1;2.5];   % Initial Condition
Tf = 100;
t  = 0:T:Tf;
N = Tf/T;

% Initialization
u  = zeros(1, N+1);
e1 = zeros(1, N+1);
e2 = zeros(1, N+1);

ym = zeros(N+1, 1);
x1 = x0(1)*ones(1, N+1);
x2 = x0(2)*ones(1, N+1);
y  = x2;
Outputs = x0;

yd = [2*ones(1, N/5), 1*ones(1,N/5), 0*ones(1, N/5), 1*ones(1,N/5), 2*ones(1,N/5+1)];

% Initial Gains and Errors
Kp = 1*ones(1, N+1);
Ki = 3*ones(1, N+1);
Kd = 0*ones(1, N+1);

ep = 0;
ei = 0;
ed = 0;

% Neural Network structure
n  = 3;      % Number of NN box Inputs for function approximation
N1 = 10;     % Number of nourons

% Initialize Weights
W1 = zeros(n, N1);
W2 = zeros(N1+1, 1);

% Main Loop
for k = 1:N
   
    Outputs = NonLinDynamic(Outputs, u(k));
    Outputs = Outputs + 0.01*norm(Outputs)*randn(size(Outputs));
    
    x1(k+1) = Outputs(1);
    x2(k+1) = Outputs(2);
    
    y(k+1)  = Outputs(2);
    
    % Forward net
    history_vector = [x1(k), x2(k), u(k)].';
    
    [ym(k+1), Phi] = f_hat(history_vector, W1, W2);
    
    e1(k+1) = y(k+1) - ym(k+1);
%     Error  = 1/2*(e1(k+1))^2
    
    % Backward Updates
    W2 = W2 + LR*(e1(k+1))* Phi;
    for i=1:n
        for j=1:N1
            W1(i,j) = W1(i,j) + LR*(e1(k+1))*W2(j)*Phi(j)*(1-Phi(j))*history_vector(i);
        end
    end
    % Update PID Control Gains
    Dym_Du = 0;
    for j=1:N1
        dym_du = W2(j)*Phi(j)*(1-Phi(j))*W1(3,j);
        Dym_Du = Dym_Du + dym_du;
    end
    
    gradE_kp = (yd(k+1)-ym(k+1))*Dym_Du*ep;
    gradE_ki = (yd(k+1)-ym(k+1))*Dym_Du*ei;
    gradE_kd = (yd(k+1)-ym(k+1))*Dym_Du*ed;
    
    % Update Control Commands
    % delta Kp,i,d(k) = -learningRate*gradient(E) + beta(t)*deltaKp,i,d(k-1)
    Kp(k+1) = Kp(k) + LR*gradE_kp - betta(k)*(Kp(k)-Kp(k));
    Ki(k+1) = Ki(k) + LR*gradE_ki - betta(k)*(Ki(k)-Ki(k));
%     Kd(k+1) = Kd(k) - LR*gradE_kd;% - betta(k)*(Kd(k)-Kd(k));
    
    % Update u(t)
    e2(k+1) = yd(k+1) - y(k+1);
    ep      = e2(k+1) - e2(k);
    ei      = T/2*(e2(k+1) + e2(k));
    if k > 1
        ed  = 1/T*(e2(k+1)-2*e2(k)+e2(k-1));
    end
    u(k+1)  = u(k) + Kp(k+1)*ep + Ki(k+1)*ei + Kd(k+1)*ed;
    
end
%% plot Outpots
figure;
plot(t, yd, '--g', 'LineWidth', 1.5)
hold on, grid on
plot(t, y, 'LineWidth', 2)
hold on, 
plot(t, ym, 'LineWidth', 2)
xlabel('Time'), ylabel('Outputs')
legend('desired', 'y', 'ym')
% axis([0 10 -5 5])

figure;
plot(t, Kp, 'LineWidth', 2)
hold on
plot(t, Ki, 'LineWidth', 2)
hold on
plot(t, Kd, 'g', 'LineWidth', 2)
xlabel('Time'), ylabel('Amp')
title('PID Gain Tuning')
legend('K_p', 'K_i', 'K_d')

%% Functions
function [ym, Phi] = f_hat(history_vector, W1, W2)

    X   = history_vector; % Inputs must be a n*1 vector
    N   = length(W2) - 1;
    phi = zeros(N,1);
    for i=1:N
        phi(i) = 1/(1+exp(-X.'*W1(:,i)));
    end
    Phi = [phi; 1];
    ym  = W2.'*Phi;       % W2 must be a (N+1)*1 vector because of bias
    
end

function out = betta(k)
    global beta0 T
    b = 1;
    out = beta0*exp(-b*k*T);
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

