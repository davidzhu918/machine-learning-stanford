function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H = X * theta;
J = (1 / (2 * m)) * sumsqr(H - y); % without regularization

new_theta = theta;
new_theta(1) = 0;

J = J + (lambda / (2 * m)) * sumsqr(new_theta); % with regularization

for j = 1:n
    grad(j) = (1 / m) * sum((H - y) .* X(:, j)) + (lambda / m) * new_theta(j, :);
end



% =========================================================================

grad = grad(:);

end
