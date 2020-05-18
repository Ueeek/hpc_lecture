#include <cstdio>
#include <immintrin.h>

int main() {
  const int N = 8;
  float a[N];
  for (int i=0; i<N; i++)a[i] = i;

  __m256 avec = _mm256_load_ps(a);//aの内容をavecに書き込む
  //[a:0 1 2 3 4 5 6 7 8]
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  //[a:0 1 2 3 4 5 6 7 8]
  //[b:4 5 6 7 8 0 1 2 3]
  bvec = _mm256_add_ps(bvec,avec);//縦にたす
  //[a:0 1 2 3 4 5 6 7 8]
  //[b: 4 6 8 10 4 6 8 10]
  bvec = _mm256_hadd_ps(bvec,bvec);//２個ずつまとめる感じ?
  //[a:0 1 2 3 4 5 6 7 8]
  //[b:10 18 10 18 10 18 10 18]
  bvec = _mm256_hadd_ps(bvec,bvec);
  //[a:0 1 2 3 4 5 6 7 8]
  //[b:28 28 28 28 28 28 28 28]
  _mm256_store_ps(a, bvec);//bvecの内容をaに書き込
  for (int i=0; i<N; i++)
    printf("%g\n",a[i]);
}
