#include<iostream>
#include<cstdio>
#include<algorithm>
#include<cstring>
#include<cmath>
#include<queue>
#include<map>
#include<set>
#define mk makr_pair
#define ll long long
using namespace std;
inline int read()
{
  int x=0,f=1;char ch=getchar();
  while (!isdigit(ch)) {if (ch=='-') f=-1;ch=getchar();}
  while (isdigit(ch)) {x=(x<<1)+(x<<3)+ch-'0';ch=getchar();}
  return x*f;
}
const int maxn = 2e5+1e2;
struct KD{
    double mx[4],mn[4],d[4];
    int l,r;
};
KD t[maxn],now;
int n,m,root;
int ymh;
double ans;
double tmp;
int ii =0;
bool operator< (KD a,KD b)
{
    return a.d[ymh]<b.d[ymh];
}
void up(int root)
{
   for (int i=0;i<=1;i++)
   {
      if (t[root].l)
      {
        t[root].mn[i]=min(t[root].mn[i],t[t[root].l].mn[i]);
        t[root].mx[i]=max(t[root].mx[i],t[t[root].l].mx[i]);
      }
      if (t[root].r)
      {
        t[root].mn[i]=min(t[root].mn[i],t[t[root].r].mn[i]);
        t[root].mx[i]=max(t[root].mx[i],t[t[root].r].mx[i]);
      }
   }
}
void build(int &x,int l,int r,int dd)
{
    ymh = dd;
    int mid = l+r >> 1;
    x = mid;
    nth_element(t+l,t+x,t+r+1);
    for (int i=0;i<=1;i++) t[x].mn[i]=t[x].mx[i]=t[x].d[i];
    if (l<x) build(t[x].l,l,mid-1,dd^1);
    if (x<r) build(t[x].r,mid+1,r,dd^1);
    up(x);
}
double getdis(int a,KD b)
{
    if (!a) return 0;
    double tmp=0;
    for (int i=0;i<=1;i++)
    {
        tmp=tmp+(t[a].d[i]-b.d[i])*(t[a].d[i]-b.d[i]);
    }
    return sqrt(tmp);
}
double calc(KD a,KD b)
{
    double tmp=0;
    for (int i=0;i<=1;i++)
    {
        if (b.d[i]<a.mn[i]) tmp=tmp+(a.mn[i]-b.d[i])*(a.mn[i]-b.d[i]);
        else if (b.d[i]>a.mx[i]) tmp=tmp+(a.mx[i]-b.d[i])*(a.mx[i]-b.d[i]);
    }
    return sqrt(tmp);
}
void query(int x)
{ 
    if (!x) return;
    double d1 = calc(t[t[x].l],now);
    double d2 = calc(t[t[x].r],now);
    double d = getdis(x,now);
    if (d<tmp && d!=0) tmp = d;
    if (d==0) ii++;
    if (d1<d2)
    {
        if (d1<tmp) query(t[x].l);
        if (d2<tmp) query(t[x].r);
    }
    else
    {
        if (d2<tmp) query(t[x].r);
        if (d1<tmp) query(t[x].l);
    }
}
int main()
{
  n=read();
  for (int i=1;i<=n;i++)
    for (int j=0;j<=1;j++) t[i].d[j]=read();
  build(root,1,n,0);
  tmp = 1e18;
  for (int i=1;i<=n;i++)
  {
   ii=0;
   now = t[i];
   query(root);
   if (ii>1) tmp=0;
  }
  printf("%.4lf",tmp);
  return 0;
}