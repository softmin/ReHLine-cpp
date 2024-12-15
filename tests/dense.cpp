#include <iostream>
#include <Eigen/Core>
#include <rehline.h>

int main()
{
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using RMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector = Eigen::VectorXd;

    // Dimensions
    const int n = 1000;
    const int p = 10;
    
    // Simulate data -- y = sign(X * beta)
    std::srand(123);
    Matrix X = Matrix::Random(n, p);
    Vector beta = Vector::Random(p);
    Vector y = X * beta;
    for (int i = 0; i < n; i++)
    {
        y[i] = (y[i] > 0.0) ? 1.0 : -1.0;
    }
    std::cout << "X[:5, :] =\n" << X.topRows(5) << std::endl;
    std::cout << "y[:5] = " << y.head(5).transpose() << std::endl;
    std::cout << "beta = " << beta.transpose() << std::endl;

    // Generate U and V according to the hinge loss (SVM)
    double C = 100.0;
    Matrix U = -C / n * y.transpose();
    Matrix V = Matrix::Constant(1, n, C / n);

    // Empty S, T, and Tau
    Matrix S(0, n), T(0, n), Tau(0, n);

    // Empty constraints
    Matrix A = Matrix::Random(0, p);
    Vector b = Vector::Random(0);

    // Setting parameters
    int max_iter = 1000;
    double tol = 1e-5;
    int shrink = 1;
    int verbose = 0;
    int trace_freq = 100;

    // Run the solver
    rehline::ReHLineResult<Matrix> res;
    rehline::rehline_solver(res, X, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);

    // Print the estimated beta
    std::cout << "\nniter = " << res.niter << "\nbeta = " << res.beta.transpose() << std::endl;

    // Make prediction and calculate accuracy rate
    Vector Xbeta = X * res.beta;
    Vector ypred = (Xbeta.array() > 0.0).select(Vector::Ones(n), -Vector::Ones(n));
    std::cout << "ypred[:5] = " << ypred.head(5).transpose() << std::endl;
    std::cout << "accuracy = " << (ypred.array() == y.array()).cast<double>().mean() << std::endl;



    // Add constraints
    const int K = 3;
    A = Matrix::Random(K, p);
    b = Vector::Random(K);

    // Run the solver again
    rehline::ReHLineResult<Matrix> res2;
    rehline::rehline_solver(res2, X, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);

    // Print results
    std::cout << "\nniter = " << res2.niter << "\nbeta = " << res2.beta.transpose() << std::endl;
    Xbeta = X * res2.beta;
    ypred = (Xbeta.array() > 0.0).select(Vector::Ones(n), -Vector::Ones(n));
    std::cout << "ypred[:5] = " << ypred.head(5).transpose() << std::endl;
    std::cout << "accuracy = " << (ypred.array() == y.array()).cast<double>().mean() << std::endl;



    // Below are just tests on different matrix types
    RMatrix RX = X, RA = A;

    // The matrix type in ReHLineResult should be consistent with (U, V, S, T, Tau)
    rehline::ReHLineResult<Matrix> res3, res4, res5;
    rehline::rehline_solver(res3, RX, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(res4, X, RA, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(res5, RX, RA, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    std::cout << "\nDifference between res2 and res3: " << (res2.beta - res3.beta).norm() << std::endl;
    std::cout << "Difference between res2 and res4: " << (res2.beta - res4.beta).norm() << std::endl;
    std::cout << "Difference between res2 and res5: " << (res2.beta - res5.beta).norm() << std::endl;

    return 0;
}
