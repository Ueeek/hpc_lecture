#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//ibodyは自分が保有するやつ。
//jbodyは、ringでぐるぐる回して計算するやつ
  Body ibody[N/size], jbody[N/size];
  srand48(rank);
  for(int i=0; i<N/size; i++) {
    ibody[i].x = jbody[i].x = drand48();
    ibody[i].y = jbody[i].y = drand48();
    ibody[i].m = jbody[i].m = drand48();
    ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = 0;
  }
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);

  //ring communicateする
    Body recv[N/size]; //rec
    MPI_Win win;
  for(int irank=0; irank<size; irank++) {
    //MPI_Send(jbody, N/size, MPI_BODY, send_to, 0, MPI_COMM_WORLD);
    //MPI_Recv(jbody, N/size, MPI_BODY, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Win_create(recv,(N/size)*sizeof(MPI_BODY),sizeof(MPI_BODY),MPI_INFO_NULL,MPI_COMM_WORLD,&win);
    MPI_Win_fence(0,win);
    MPI_Put(jbody,N/size,MPI_BODY, send_to,0,N,MPI_BODY,win);
    MPI_Win_fence(0,win);
    for(int i=0;i<N/size;i++)jbody[i]=recv[i]; //copy recv into jbody
    
    for(int i=0; i<N/size; i++) {
      for(int j=0; j<N/size; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        //ringにしているから、i==jでは同じ物体と判断できない。
        //二つの距離が十分に小さいなら同一とみなす
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }
  }
    MPI_Win_free(&win);
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        printf("%d %g %g\n",i+rank*N/size,ibody[i].fx,ibody[i].fy);
      }
    }
  }
  MPI_Finalize();
}
