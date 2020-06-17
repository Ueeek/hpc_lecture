#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sys/time.h>
#include<openacc.h>



double get_elapsed_time(struct timeval *begin, struct timeval *end){
    return (end->tv_sec -  begin->tv_sec)*1000000
        +(end->tv_usec-begin->tv_usec);
}

//global variables
int nx = 101;
int ny =101;
int nt =1000;
int nit = 100;

float c = 1.0;
float dx = 2.0/(nx-1);
float dy = 2.0/(ny-1);

float rho=1.0;
float nu = 0.1 ;
float dt = 0.001;

//write out data into txt file in order to plot this result with matplot lib in a notebook
template<typename T>
void write_ouput_file(std::string name,T* v,int H,int W){
    std::ofstream outputfile(name);
    for(int i=0;i<H;i++){
        for(int j=0;j<W;j++){
            outputfile<<v[i*W+j]<<" ";
        }
        outputfile<<"\n";
    }
    outputfile.close();
}

//print 1D array as 2D array
template<typename T>
void print_vec_as_2D(T* v,int H,int W){
    for(int i=0;i<H;i++){
        for(int j=0;j<W;j++){
            printf("%f ",v[i*W+j]);
        }
        std::cout<<"\n";
    }
}

void build_up_b(float* b, float* u, float* v){
#pragma acc kernels
    {
#pragma acc  loop independent
    for(int i=1;i<ny-1;i++){
#pragma acc  loop seq
        for(int j=1;j<nx-1;j++){
            b[i*nx+j] = (rho*(1.0/dt
                            *((u[i*nx+j+1] - u[i*nx+j-1])
                            /(2.0*dx)+(v[(i+1)*nx+j]-v[(i-1)*nx+j])/(2.0*dy))
                            - ((u[i*nx+j+1] - u[i*nx+j-1]) / (2.0*dx))*((u[i*nx+j+1] - u[i*nx+j-1])/(2.0*dx))
                            - 2.0*((u[(i+1)*nx+j]-u[(i-1)*nx+j])/(2.0*dy)
                            *(v[i*nx+j+1]-v[i*nx+j-1])/(2.0*dx))
                            -((v[(i+1)*nx+j]-v[(i-1)*nx+j])/(2.0*dy))*((v[(i+1)*nx+j]-v[(i-1)*nx+j])/(2.0*dy))));
        }
    }
    }//kernels
}


void pressure_poisson(float* p, float* b){
    float pn[nx*ny];
#pragma data copy(pn[0:nx*ny])
    {
#pragma acc kernels
        {
#pragma acc  loop independent
    for(int i=0;i<nx*ny;i++) pn[i]= p[i];//copy from p

#pragma acc  loop seq
    for(int q=0;q<nit;q++){
#pragma acc  loop independent
        for(int i=0;i<nx*ny;i++) pn[i]= p[i];//copy from p

#pragma acc  loop seq
        for(int i=1;i<ny-1;i++){
#pragma acc  loop independent
            for(int j=1;j<nx-1;j++){
                p[i*nx+j] = (((pn[i*nx+j+1]+pn[i*nx+j-1])*dy*dy
                            + (pn[(i+1)*nx+j]+pn[(i-1)*nx+j])*dx*dx)
                            /(2.0*(dx*dx+dy*dy))
                            -dx*dx*dy*dy
                            /(2.0*(dx*dx+dy*dy))
                            * b[i*nx+j]);
            }
        }

#pragma acc loop independent
        for(int i=0;i<ny;i++){
            p[i*nx+nx-1] = p[i*nx+nx-2];
            p[i*nx+0] = p[i*nx+1];
        }
#pragma acc  loop independent
        for(int i=0;i<nx;i++){
            p[0*nx+i] = p[1*nx+i];
            p[(ny-1)*nx+i] =0.0;
        }
    }
    }//kernels
    }//data
}

void cavaty_flow(float* u, float* v, float* p){
    float un[ny*nx];
    float vn[ny*nx];
    float b[ny*nx];

#pragma acc kernels
#pragma acc  loop independent
    for(int i=0;i<nx*ny;i++)b[i]=0.0;

//#pragma acc loop seq
#pragma acc data copy(u[0:nx*ny],v[0:nx*ny],p[0:nx*ny],un[0:nx*ny],vn[0:nx*ny],b[0:nx*ny])
    {
    for(int t=0; t<nt;t++){
#pragma acc kernels
        {
#pragma acc loop independent
        for(int i=0;i<nx*ny;i++) un[i] = u[i];
#pragma acc loop independent
        for(int i=0;i<nx*ny;i++) vn[i] = v[i];
        }//kernels

        //sequencial ( parallelize inside each function
        build_up_b(b,u,v);
        pressure_poisson(p,b);


#pragma acc kernels
        {
#pragma acc  loop seq
        for(int i=1;i<ny-1;i++){
#pragma acc  loop seq
            for(int j=1;j<nx-1;j++){
                u[i*nx+j] = (un[i*nx+j]-un[i*nx+j]*dt/dx
                        *(un[i*nx+j]-un[i*nx+j-1])
                        - vn[i*nx+j]*dt/dy
                        * (un[i*nx+j] - un[(i-1)*nx+j])
                        - dt/(2.0*rho*dx)*(p[i*nx+j+1]-p[i*nx+j-1])
                        + nu*(dt/dx/dx *(un[i*nx+j+1]-2.0*un[i*nx+j]+un[i*nx+j-1])
                        +dt/dy/dy*(un[(i+1)*nx+j]-2.0*un[i*nx+j]+un[(i-1)*nx+j])));

                v[i*nx+j] = (vn[i*nx+j]-un[i*nx+j]*dt/dx
                        *(vn[i*nx+j]-vn[i*nx+j-1])
                        - vn[i*nx+j]*dt/dy
                        * (vn[i*nx+j] - vn[(i-1)*nx+j])
                        - dt/(2.0*rho*dy)*(p[(i+1)*nx+j]-p[(i-1)*nx+j])
                        + nu*(dt/dx/dx *(vn[i*nx+j+1]-2.0*vn[i*nx+j]+vn[i*nx+j-1])
                        +dt/dy/dy*(vn[(i+1)*nx+j]-2.0*vn[i*nx+j]+vn[(i-1)*nx+j])));
            }
        }
#pragma acc  loop independent
        for(int i=0;i<ny;i++){
            u[i*nx+0]=0.0;
            u[i*nx+nx-1]=0.0;
            v[i*nx+0]=0.0;
            v[i*nx+nx-1]=0.0;
        }

#pragma acc  loop independent
        for(int i=0;i<nx;i++){
            u[0*nx+i]=0.0;
            u[(ny-1)*nx+i]=1.0;
            v[0*nx+i]=0.0;
            v[(ny-1)*nx+i]=0.0;
        }
    }//kernels
    }//loop of t
    }//data
}


int main(void){
    //np.linspace
    float x[nx],y[ny];
    for(int i=0;i<nx;i++) x[i]=dx*i;
    for(int i=0;i<ny;i++) y[i]=dx*i;

    //np.meshgrid
    float X[nx*ny],Y[nx*ny];

    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            X[i*nx+j] = x[j];
            Y[i*nx+j] = y[i];
        }
    }
    //np.zeros
    float u[ny*nx];
    float v[ny*nx];
    float p[ny*nx];
    float b[ny*nx];
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            u[i*nx+j]=0.0;
            v[i*nx+j]=0.0;
            p[i*nx+j]=0.0;
            b[i*nx+j]=0.0;
        }
    }
    struct timeval start,end;

    gettimeofday(&start,NULL);
    cavaty_flow(u,v,p);
    gettimeofday(&end,NULL);

    double us;
    us = get_elapsed_time(&start,&end);
    printf("Elapsed time : %.3lf sec\n",us/1000000.0);



    //write_ouput_file("./res/p.txt",p,ny,nx);
    //write_ouput_file("./res/u.txt",u,ny,nx);
    //write_ouput_file("./res/v.txt",v,ny,nx);
    //write_ouput_file("./res/X.txt",X,ny,nx);
    //write_ouput_file("./res/Y.txt",Y,ny,nx);
    return 0;
}
