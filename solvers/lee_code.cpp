struct eigen_solver{
    //Eigen solver comparison
    eigen_solver(const int solver_Eigen, const int n_rows, const int n_cols,
              const scitbx::af::shared<intType> A_row_offset,
              const scitbx::af::shared<intType> A_col_idx,
              const scitbx::af::shared<numType> A_values,
              const scitbx::af::shared<numType> b) : x_eig(n_rows, 0.){

      //Eigen::SparseMatrix<double> spMat(n_rows,n_cols);
      Eigen::SparseMatrix<double, Eigen::RowMajor> spMat(n_rows,n_cols);
            std::vector<Eigen::Triplet<double,int> > tList;

      Eigen::VectorXd b_internal(n_cols);
      for (int i = 0; i<n_cols; ++i){
         b_internal[i] = *(b.begin()+i);
      }

      int c_l, r_i;
      for( int row = 0; row < n_cols; ++row ){
        r_i = *(A_row_offset.begin() + row); //Row value at index i
        c_l = ( *( A_row_offset.begin() + 1 + row ) - r_i ); //Column length between the given row offsets
        for( int col = 0; col < c_l; ++col ){
           //std::cout << "ROW=" << row << " COL=" <<  *(A_col_idx.begin() + col + r_i) << " VAL=" << *(A_values.begin() + col + r_i) << std::endl;
           tList.push_back(
             Eigen::Triplet<double,int>( row, *(A_col_idx.begin() + col + r_i), *(A_values.begin() + col + r_i) )
           );
        }
      }
      spMat.setFromTriplets( tList.begin(), tList.end() ); //Form the Eigen matrix from the given (i,j,value) sparse format
      spMat.makeCompressed();

      Eigen::VectorXd x(n_rows);

      if(solver_Eigen == 0){
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > chol(spMat.transpose());
        x = chol.solve(b_internal);
      }
      else if(solver_Eigen == 1){
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > chol(spMat.transpose());
        x = chol.solve(b_internal);
      }
      else if(solver_Eigen == 2){
        Eigen::SparseMatrix<double> eigen_normal_matrix_full = spMat.selfadjointView<Eigen::Upper>();
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor> > solver(eigen_normal_matrix_full);
        x = solver.solve(b_internal);
      }
      else {
        Eigen::SparseMatrix<double> eigen_normal_matrix_full = spMat.selfadjointView<Eigen::Upper>();
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver(eigen_normal_matrix_full);
        x = solver.solve(b_internal);
      }
      double* solnptr = x_eig.begin();
      for (int i = 0; i<n_rows; ++i){
        *solnptr++ = x[i];
      }
    }
    scitbx::af::shared<numType> x_eig; //Resulting x will be stored here
  }; //end of struct sparse_solver