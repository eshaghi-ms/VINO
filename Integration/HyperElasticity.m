% HyperElasticity Beam

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% U_x,0 = U_11 - U_00
%%% U_x,1 = U_11 - U_01
%%% U_y,0 = U_01 - U_00
%%% U_y,1 = U_11 - U_10 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear

syms nu x y n real
syms dUdx_0 dUdx_1 dUdy_0 real
syms dVdx_0 dVdx_1 dVdy_0 real
syms dx dy real
syms c d c1 c2 real

paramb1 = dUdx_0/dx;
paramc1 = dUdy_0/dy;
paramd1 = (dUdx_1-dUdx_0)/(dx*dy);

paramb2 = dVdx_0/dx;
paramc2 = dVdy_0/dy;
paramd2 = (dVdx_1-dVdx_0)/(dx*dy);


u = paramb1 * x + paramc1 * y + paramd1 * x * y;
v = paramb2 * x + paramc2 * y + paramd2 * x * y;


dUdx = diff(u, x);
dUdy = diff(u, y);
dVdx = diff(v, x);
dVdy = diff(v, y);

Fxx = dUdx + 1;
Fxy = dUdy + 0;
Fyx = dVdx + 0;
Fyy = dVdy + 1;
detF = Fxx * Fyy - Fxy * Fyx;
C11 = Fxx * Fxx + Fyx * Fyx;
C12 = Fxx * Fxy + Fyx * Fyy;
C21 = Fxy * Fxx + Fyy * Fyx;
C22 = Fxy * Fxy + Fyy * Fyy;
J = detF;
traceC = C11 + C22;
I1 = traceC;
trace_C2 = C11 * C11 + C12 * C21 + C21 * C12 + C22 * C22;
I2 = 0.5 * (traceC^2 - trace_C2);

% ln = @(j) (j - 1) - (1/2) * (j - 1)^2 + (1/3) * (j - 1)^3 - (1/4) * (j - 1)^4;
% ln = @(j) (j - 1) - (1/2) * (j - 1)^2;
ln = @(j) (j - 1);

strainEnergy = c * (J - 1) ^ 2 - d * ln(J) + c1 * (I1 - 2) + c2 * (I2 - 1);

expr = strainEnergy;

integrated_expr = int(int(expr, x, 0, dx), y, 0, dy);
disp(integrated_expr);
simplifiedExpr = simplifyFraction(integrated_expr);
disp(simplifiedExpr);

%dy = n*dx
expr_2 = simplifyFraction(subs(simplifiedExpr, dy, n*dx));
disp(expr_2);

% n = 1
%expr_3 = simplifyFraction(subs(expr_2, n, 1));
%disp(expr_3);

