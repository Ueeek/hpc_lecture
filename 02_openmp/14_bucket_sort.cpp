#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

std::vector<int> bucket(range); 
#pragma omp for
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
#pragma opm for
      for(int k=0;k<bucket[i];k++){
          key[j+k]=i;
      }
      j+=bucket[i];
    //for (; bucket[i]>0; bucket[i]--) {
    //  key[j++] = i;
    //}
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
