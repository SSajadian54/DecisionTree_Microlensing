#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include "VBBinaryLensingLibrary.h"
using namespace std;
///===============================================
    time_t _timeNow;
    unsigned int _randVal;
    unsigned int _dummyVal;
    FILE * _randStream;
//================================================


int main(){
//================================================
   time(&_timeNow);
   _randStream = fopen("/dev/urandom", "r");
   _dummyVal = fread(&_randVal, sizeof(_randVal), 1, _randStream);
   srand(_randVal);
   _dummyVal = fread(&_randVal, sizeof(_randVal), 1, _randStream);
   srand(_randVal);
///=================================================
   printf("START_TIME >>>>>>>> %s",ctime(&_timeNow));

   FILE* fil2;  FILE*  fil1;  FILE* fil3; 
   fil2=fopen("./efsf1.csv","a+");
   fil3=fopen("./param1.txt","a+");
   //fprintf(fil2,"mbase, Deltam, FWHM, Tmax, fws, rho, tstar, ur,  fb,  mstar\n");

   VBBinaryLensing vbb;
   vbb.Tol=1.e-5;
   vbb.a1=0.0;
   vbb.LoadESPLTable("./ESPL.tbl");

   char filnam[40]; 
   double tmax1, maxm1; 
   double tim1, tim2, pt1, pt2;  
   double tstar, u, tim, Astar, thre; 
   double u0, tE, rho, fb,  mstar, tmm; 
   double twing,  tshol, fws,  mbase,  magm, FWHM;
   int flag2, flag1, N, flagf, error; 
   double dt;//days 
   
   
   
   
   for(int icon=0; icon<600000; ++icon){
  
   do{
   rho=double((double)rand()/(double)(RAND_MAX+1.))*9.0+1.0;
   u0=double((double)rand()/(double)(RAND_MAX+1.))*10.0; 
   }while(u0>fabs(rho));
   tE=double((double)rand()/(double)(RAND_MAX+1.))*4.0+3.0;
   fb=double((double)rand()/(double)(RAND_MAX+1.))*1.0;
   mstar=double((double)rand()/(double)(RAND_MAX+1.))*4.0+16.0;

   
   mbase= double(mstar+2.5*log10(fb));
   tstar= double(rho*tE); 
   dt=double(tstar/5000.0); 
   
   pt1= -5.0*tstar;
   pt2= 0.0;///+5.0*tstar; 
   N= int((pt2-pt1)/dt)+1;
   cout<<"N:  "<<N<<endl; 
   double mag[N], der1[N];
   flag1=flag2=0; 
   maxm1=tmax1=0.0; 
   FWHM=tim1=tim2=magm=0.0; 
   twing=tshol=fws=0.0; 
   for(int i=0; i<N; ++i) mag[i]=der1[i]=0.0;
   
   
   
   for(int i=0; i<N; ++i){
   tim=double(pt1 + i*dt); //days
   u=sqrt(u0*u0 + tim*tim/tE/tE);
   Astar=vbb.ESPLMag2(u , rho);
   if(Astar<1.0){cout<<"Error Astar:  "<<Astar<<"\t u:  "<<u<<"\t  rho:  "<<rho<<endl;  Astar=1.0; } 
   mag[i]=double(mbase-2.5*log10(fb*Astar+1.0-fb) );//magnitude
   if(i>0){
   der1[i]=fabs(mag[i]-mag[i-1])/dt;
   if(der1[i]==0.0){
   cout<<"i:  "<<i<<"\t deri[i]:   "<<der1[i]<<endl;   
   der1[i]= 0.0000000006346364*double((double)rand()/(double)(RAND_MAX+1.)); }}
   if(der1[i]>maxm1 and tim<0.0){maxm1= der1[i];      tmax1=tim;}//Time with Max_Deri1  
   if(fabs(mag[i]-mbase)>magm)  magm=fabs(mag[i]-mbase);}
   
   
   ///*****************************************************************************************
   thre= 0.25*fabs(log10(maxm1)- log10(der1[2]) ); 
   thre= maxm1*pow(10.0 , -thre);
   flag1=flag2=0; 
   for(int i=1; i<N; ++i){ 
   tim= double(pt1 + i*dt);///days
   if(fabs(mag[i]-mbase)>fabs(0.5*magm) and flag2==0){FWHM=fabs(2.0*tim);    flag2=1;  tmm=tim;}  
   if(der1[i]>thre and flag1==0 and tim<0.0)         {tim1=tim;               flag1=1;}
   if(der1[i]<thre and flag1==1 and tim<0.0 and fabs(tim)<fabs(tmax1)){tim2=tim;  flag1=2;}}
   twing=fabs(fabs(tim2)- fabs(tim1))*1.0;//wings
   tshol= -tim2;//sholders
   fws= fabs(twing/tshol); 
   
  
  
   if(tshol<0.0 or fabs(tim2)>fabs(tmax1)  or fabs(tim1)<fabs(tmax1) or fabs(twing+tshol)>fabs(pt1) or FWHM<0.0 or FWHM==0.0){
   error=1; 
   cout<<"Error icon:      "<<icon<<"\t u0/rho:  "<<u0/rho<<"\t fb:  "<<fb<<endl;
   cout<<"tim1/tstar:  "<<tim1/tstar<<"\t tmax1/tstar:   "<<tmax1/tstar<<"\t tim2/star:  "<<tim2/tstar<<endl;
   cout<<"pt1:  "<<fabs(pt1/tstar)<<"\t twing:   "<<twing/tstar<<"tsholder:  "<<tshol/tstar<<endl;
   sprintf(filnam,"./files/%c%c%d.dat",'m','_',icon);
   fil1=fopen(filnam,"w");
   for(int j=0; j<N; ++j){
   tim= double(pt1 + j*dt); 
   fprintf(fil1,"%.5lf  %.10lf   %.10lf\n",tim/tstar, mag[j], der1[j]);}
   fclose(fil1);}
   else{
   error=-1;
   fprintf(fil2,"%.8lf,  %.8lf, %.8lf, %.8lf,  %.8lf,  %.8lf,  %.8lf,  %.8lf,  %.8lf,  %.8lf\n",
   mbase, fabs(magm), fabs(FWHM), fabs(tmax1), fws, rho, tstar, u0/rho, fb, mstar);}
    
    
    
   fprintf(fil3,"%d   %.6lf  %.6lf %.6lf %.6lf  %.6lf  %.6lf  %.6lf  %.6lf  %.6lf  %.6lf  %.6lf  %.6lf  %.6lf  %.6lf  %.6lf %d  %d\n",
   icon, mbase, fabs(magm), fabs(FWHM), fabs(tmax1), fws, rho, tstar, u0/rho, fb, mstar, tim1, tmm, tim2, twing, tshol, error,N);
   cout<<"================================================"<<endl;  
   cout<<"*************>>>   "<<icon<<"  <<< ***************"<<endl; 
   cout<<"Maximum_Delta_magnitude   "<< magm<<"\t error:  "<<error<<endl;
   cout<<"maxm1:  "<<maxm1<<"\t tmax1: "<<tmax1/tstar<<endl;  
   cout<<"FWHM:   "<<FWHM<<"\t  flag:  "<<flag2<<endl;
   cout<<"twing:  "<<twing/tstar<<"\t tshold:  "<<tshol/tstar<<"\t fws:  "<<fws<<endl;
   cout<<"mbase:  "<<mbase<<"\t tE:  "<<tE<<endl;
   cout<<"rho:  "<<rho<<"    \t u0:  "<<u0<<endl;
   cout<<"fb: "<<fb<<"\t mag_star:  "<<mstar<<endl;
   cout<<"tim1:  "<<fabs(tim1+fabs(tmax1))/tstar<<"\t tim2:  "<<fabs(tim2+fabs(tmax1))/tstar<<"\t tmax1:  "<<tmax1/tstar<<endl;
   cout<<"tim1:   "<<tim1/tstar<<"\t tim2:   "<<tim2/tstar<<endl;
   cout<<"threshold:    "<<pow(10.0,-thre)<<endl;
   cout<<"time(u=rho):   "<<tstar*sqrt(1.0-(u0*u0)/(rho*rho) )<<endl;
   cout<<"tmax1 "<<fabs(tmax1) <<"\t  FWHM "<<fabs(FWHM/2.0)<<endl;
   cout<<"================================================"<<endl;  }//end of loop icon
   
   
   fclose(fil2); 
   fclose(fil3); 
   printf("END_TIME >>>>>>>> %s",ctime(&_timeNow));
   return(0); 
}
