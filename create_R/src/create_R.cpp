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
                if(h==0) cout<<".";
                else cout<<"O";
                //cout<<(h==0)?"A":"O";
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
    vector<vector<int>> rotated(13);  //Vector need to have clear boundaries, and it is important to focus on learning how to express vector.
    for (int i=0;i<13;i++)
         rotated[i].resize(13);
     //If the source document has only 9 lines, setting it to 13 lines here will have no effect.
    for (int i=0;i<13;i++)
         matrix[i].resize(13);

    for (int i = 0; i < 13; i++)
    {
        for (int j = 0; j < 13; j++)
        { 
            rotated[i][12-j] = matrix[j][i];  
        }
    }
    
     
   for (int i = 0; i < 13; i++)
    {
        for (int j = 0; j < 13; j++)
        {
            matrix[i][j]= rotated[i][j];// 
        }    
    }
    
   // for (int i=0;i<15;i++)
   //      matrix[i].resize(13); 
             
}           
    
  
};
int main()
{
     matrix_image img;
      cout<<"read_restore result is:";
      cout<<endl;
      img.read_restore("/home/stark/create_R/src/matrix_finalpic.txt",img.matrix);
      img.display(9,13); 
        
        cout<<"thresholding result is:";
        cout<<endl;
        img.thresholding(100); 
        img.display(9,13);

        cout<<"binary_print result is:"; 
        cout<<endl;
        img.binary_print();

        cout<<"flip result is:";
        cout<<endl;
        img.flip();
        img.display(9,13);

        cout<<"binary_print result is:"; 
        cout<<endl;
        img.binary_print();

        for(int k=0;k<3;k++)
        {
           cout<<"rotate result is:";
           cout<<endl;
           img.rotate();
           img.display(13,9); 


           cout<<"binary_print result is:"; 
           cout<<endl;
           img.binary_print();
        }

    for (int i=0;i<15;i++)
        img.matrix[i].resize(9);   

    cout<<"binary_print result is:"; 
    cout<<endl;
    img.binary_print();

    return 0;     
}