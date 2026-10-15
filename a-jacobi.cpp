









//assuming all other hosts pointers for now. There will be certain memcpy for collision and non collision cases
bool a_jacobi(int n, int max_iter, float abs_tol, float rel_tol,
              const CSRMatrix& B, const float* D,
              const float* b, float* x){

            // D^-1 * b
            float* d_inv_b = new float[B.col_num];
            for(int i=0;i<B.col_num;i++){
                d_inv_b[i] = b[i]/D[i];
            }
            
            



            }
    



        
            
