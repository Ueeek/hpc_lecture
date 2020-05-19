#include <cstdio>
#include <cstdlib>
#include <cmath>
#include<immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 x_vec = _mm256_load_ps(x);
  __m256 y_vec = _mm256_load_ps(y);
  __m256 m_vec = _mm256_load_ps(m);
  for(int i=0; i<N; i++){
    //mask
    float MASK[N];
    for(int m=0;m<N;m++)MASK[m]=m;

    __m256 i_vec = _mm256_set1_ps(i);
    __m256 x_i_vec = _mm256_set1_ps(x[i]);
    __m256 y_i_vec = _mm256_set1_ps(y[i]);

    __m256 j_vec = _mm256_load_ps(MASK);
    __m256 mask = _mm256_cmp_ps(j_vec,i_vec,_CMP_NEQ_OQ);

    //apply mask
    __m256 x_j_vec = _mm256_setzero_ps();
    __m256 y_j_vec = _mm256_setzero_ps();
    __m256 m_j_vec = _mm256_setzero_ps();
    
    x_j_vec = _mm256_blendv_ps(x_j_vec,x_vec,mask);
    y_j_vec = _mm256_blendv_ps(y_j_vec,y_vec,mask);
    m_j_vec = _mm256_blendv_ps(m_j_vec,m_vec,mask);


    //x[i]-x[j]
    __m256 rx_vec = _mm256_sub_ps(x_i_vec,x_j_vec);
    __m256 ry_vec = _mm256_sub_ps(y_i_vec,y_j_vec);


    //r = sqrt(rx*rx+ry*ry)
    __m256 rx_2_vec = _mm256_mul_ps(rx_vec,rx_vec);
    __m256 ry_2_vec = _mm256_mul_ps(ry_vec,ry_vec);
    
    __m256 r_double_vec = _mm256_add_ps(rx_2_vec,ry_2_vec);
    __m256 r_vec = _mm256_rsqrt_ps(r_double_vec);

    
    //1/r*r*r
    __m256 r2_vec = _mm256_mul_ps(r_vec,r_vec);
    __m256 r3_vec = _mm256_mul_ps(r_vec,r2_vec);

    //fx,fy
    __m256 rxm_vec = _mm256_mul_ps(rx_vec,m_j_vec);
    __m256 rym_vec = _mm256_mul_ps(ry_vec,m_j_vec);

    __m256 i_x_fvec = -_mm256_mul_ps(rxm_vec,r3_vec);
    __m256 i_y_fvec = -_mm256_mul_ps(rym_vec,r3_vec);


    //aggregate
    __m256 x_fvec = _mm256_permute2f128_ps(i_x_fvec,i_x_fvec,1);
    x_fvec = _mm256_add_ps(x_fvec,i_x_fvec);
    x_fvec = _mm256_hadd_ps(x_fvec,x_fvec);
    x_fvec = _mm256_hadd_ps(x_fvec,x_fvec);

    __m256 y_fvec = _mm256_permute2f128_ps(i_y_fvec,i_y_fvec,1);
    y_fvec = _mm256_add_ps(y_fvec,i_y_fvec);
    y_fvec = _mm256_hadd_ps(y_fvec,y_fvec);
    y_fvec = _mm256_hadd_ps(y_fvec,y_fvec);

    //write
    _mm256_store_ps(fx,x_fvec);
    _mm256_store_ps(fy,y_fvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
