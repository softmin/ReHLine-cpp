#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <rehline.h>

int main()
{
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using RMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector = Eigen::VectorXd;
    using SpMat = Eigen::SparseMatrix<double>;
    using SpRMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

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

    // Add constraints
    const int K = 3;
    Matrix A = Matrix::Random(K, p);
    Vector b = Vector::Random(K);

    // Sparse matrices
    SpMat Xsp = X.sparseView(), Asp = A.sparseView();

    // Setting parameters
    int max_iter = 1000;
    double tol = 1e-5;
    int shrink = 1;
    int verbose = 0;
    int trace_freq = 100;

    // Run the solver
    rehline::ReHLineResult<Matrix> res;
    rehline::rehline_solver(res, Xsp, Asp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);

    // Print the estimated beta
    std::cout << "\nniter = " << res.niter << "\nbeta = " << res.beta.transpose() << std::endl;

    // Make prediction and calculate accuracy rate
    Vector Xbeta = Xsp * res.beta;
    Vector ypred = (Xbeta.array() > 0.0).select(Vector::Ones(n), -Vector::Ones(n));
    std::cout << "ypred[:5] = " << ypred.head(5).transpose() << std::endl;
    std::cout << "accuracy = " << (ypred.array() == y.array()).cast<double>().mean() << std::endl;



    // Below are just tests on different matrix types
    RMatrix RX = X, RA = A;
    SpRMat RXsp = Xsp, RAsp = Asp;
    // The matrix type in ReHLineResult should be consistent with (U, V, S, T, Tau)
    rehline::ReHLineResult<Matrix> more_res[11];
    // Mixing row- and column-majored sparse matrices
    rehline::rehline_solver(more_res[0], RXsp, Asp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[1], Xsp, RAsp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[2], RXsp, RAsp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    // Mixing dense and sparse matrices
    rehline::rehline_solver(more_res[3], X, Asp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[4], X, RAsp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[5], RX, Asp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[6], RX, RAsp, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[7], Xsp, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[8], RXsp, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[9], Xsp, RA, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    rehline::rehline_solver(more_res[10], RXsp, RA, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq);
    // Test results
    std::cout << std::endl;
    for (int i = 0; i < 11; i++)
    {
        std::cout << "Difference between res and more_res[" << i << "]: " << (res.beta - more_res[i].beta).norm() << std::endl;
    }

    return 0;
}
