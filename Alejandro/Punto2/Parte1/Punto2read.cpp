#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
using namespace std;

const int Angle=60;
const double slope=1.4;
const double intercept=294;

void Divide(const char * MyFile){
  ifstream infile; ofstream outfile; int n=0; 
  double ix, iy;
  for(int i=1;i<5;i++){
    infile.open(MyFile);
    outfile.open(to_string(i)+"Datos"+to_string(Angle)+".dat");
    while(infile>>ix>>iy){
      if(slope*(ix-(intercept-i*6.0)-3.0)+iy<0.0 && slope*(ix-(intercept-i*6.0)+3.0)+iy>0.0){ 
        outfile<<ix<<" "<<iy<<endl;
      }
    }
    outfile.close();
    infile.close();
  }
}
 
int main(void){
  Divide("LongitudDeOnda60.dat");  
  return 0;
}
