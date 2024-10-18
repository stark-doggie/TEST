#include<algorithm>
#include<fstream> 
#include<sstream>
#include<string>
#include<cmath>
#include<vector>
#include<iostream>
using namespace std;

class matrix_image{
    
           
    public: 
          vector<vector<int>> matrix;
           int W,H;
         
void init_matrix(const vector<vector<int>> matrix)
{
    int w,h;
    cin>>w>>h; 
     for(auto &w:matrix)
        for(auto &h:w)
            {
                cout<<0<<endl;
            }   
}


void read_restore(const string &filename,vector<vector<int>> &matrix)
{
       ifstream infile(filename);//wait to be improved
       if(!infile){
            cout<<"document can not be opened!"<<endl;
            return;
       }
       matrix.clear();
       string line;
       while(getline(infile,line)){
          vector<int> W;
          int value;
          istringstream iss(line);
          while(iss>>value)
          {
            W.emplace_back(value);
            //cout<<value;
          }
          matrix.emplace_back(W); 
       }
       W=matrix.size();
        
       H=matrix.empty() ? 0 : matrix[0].size();
        
       infile.close();
}



bool judge_empty(int H,int W)
{
     
    if((H==0)&&(W==0))
    {
        cout<<"is empty!"<<endl;
        return false;
    }  
    return true;
}


void display_point(int X,int Y,int VALUE,vector<vector<int>> matrix)
{
     matrix[X][Y]=VALUE;
}


void binary_print()
{
     for(auto &w:matrix){
        for(auto &h:w)
            {
                cout<<(h==0)?".":"O";
                cout<<" ";
                
            };
        cout<<endl; 
     }
}


void display_rect(vector<vector<int>> matrix)
{
    int X,Y,W,H,VALUE; 
    for(X=0;X<W;X++)
       for(Y=0;Y<H;Y++)
          {
           cin>>VALUE;
           matrix[X][Y]=VALUE;
           cout<<matrix[X][Y];
          };
}


void thresholding(int THR)
{
   for(auto &w:matrix)
        for(auto &h:w){
            if(h<THR)
               h=0;
        }; 
}


void flip()
{
    for(auto &w:matrix)
        reverse(w.begin(),w.end());
}

void display(int h,int w)
{
     
    for(auto &w:matrix)
        {
        for(auto &h:w)
            {
                cout<<h<<" ";
            }; 
        cout<<endl;
        }
}

void rotate()
{
      vector<vector<int>> rotated(7);  
      for (int i=0;i<7;i++)
         rotated[i].resize(7);
      for (int i=0;i<7;i++)
         matrix[i].resize(7);

    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
           
             rotated[i][6- j] = matrix[j][i];  
        }
    }
    
     
    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            matrix[i] [j]= rotated[i][j+2]; 
        }    
    }
    
    for (int i=0;i<7;i++)
         matrix[i].resize(5); 
             
}           

 
     
  
};
int main()
{
     matrix_image img;
      cout<<"read_restore result is:";
      cout<<endl;
      img.read_restore("/home/stark/matrix_zuo/src/matrix_5x7pic.txt",img.matrix);
      img.display(5,7); 
        
        cout<<"thresholding result is:";
        cout<<endl;
        img.thresholding(5); 
        img.display(5,7);

        cout<<"flip result is:";
        cout<<endl;
        img.flip();
        img.display(5,7);
 
      cout<<"rotate result is:";
      cout<<endl;
      img.rotate();
      img.display(5,7);

     cout<<"binary_print result is:"; 
     cout<<endl;
      img.binary_print();
 
    return 0;     
}