function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
    yi = y(i,:);
    xi = X(i,:);
    hi = sigmoid(theta' * xi');
    J = J + (yi .* log(hi) + (1 - yi) .* log(1 - hi));
end
J = -1 .* J ./ m;

sum = 0;
for j = 2 : n
    sum = sum + theta(j) .^ 2;
end
J = J + (lambda .* sum) ./ (2 .* m);


grad = (X' * (sigmoid(X * theta) - y)) ./ m;
for i = 2 : n
    grad(i) = grad(i) + (lambda .* theta(i)) ./ m;
end


% =============================================================

end
