# Error Analysis

Analyzed `outputs/predictions/hybrid_rerank_predictions.jsonl` with 12000 query-level best hybrid predictions.

## High lexical similarity but different semantics

- `problem_4` vs `problem_52`, TF-IDF score `1.0000`: `int main() { int m,n,i,j,k,a[100][100]; scanf("%d%d",&m,&n); for(i=0;i<m;i++) for(j=0;j<n;j++) { scanf("%d",&a[i][j]); } printf("%d\n",a[0][0]); for(j=1;j<n;j++) { for(i=0;i<j+1&&i<m;i++) { printf("%d\n",a[i][j-i]); } } ...` / `int main() {int n, m, a[100], b[100], *p, i, j, t, k; scanf("%d%d",&n,&m); for(i=0;i<n;i++) {scanf("%d",&a[i]);} p=&a[n-m]; for(i=n-m;i<n;i++) {b[i-n+m]=*(p++);} for(j=0;j<m;j++) { for(i=j,t=a[i];i<n;i++) { k=a[i+1]; a[i...`
- `problem_10` vs `problem_15`, TF-IDF score `1.0000`: `void main() {int a[100],b[100],i,j,k=0,n; scanf("%d",&n); for(i=0;i<n;i++) {scanf("%d",&a[i]); b[i]=1;} for(i=n-1;i>=0;i--) {k=0; for(j=n-1;j>i;j--) {if(a[i]>=a[j]&&b[j]>k) {k=b[j]; b[i]=b[j]+1;} } } k=b[0]; for(i=0;i<n;...` / `void main() { int a,b,c,d,n,i,j,x,y,s,m[100][100]; scanf("%d",&n); for(i=0;i<n;i++) for(j=0;j<n;j++) scanf("%d",&m[i][j]); for(i=0;i<n;i++) for(j=0;j<n;j++) { if(m[i][j]==0 && m[i+1][j]==0 && m[i][j+1]==0) { a=i;b=j; } i...`
- `problem_13` vs `problem_53`, TF-IDF score `1.0000`: `int main() { int a[20000],n,i,j; scanf("%d",&n); for(i=0;i<n;i++) { scanf("%d",&a[i]); } printf("%d",a[0]); for(i=1;i<n;i++) { for(j=0;j<i;j++) { if(a[i]==a[j]) {j=i;} else {j=j;} } if(j==i) printf(" %d",a[i]); } return ...` / `int main() { int a[20000],n,i,j; scanf("%d",&n); for(i=0;i<n;i++) { scanf("%d",&a[i]); } printf("%d",a[0]); for(i=1;i<n;i++) { for(j=0;j<i;j++) { if(a[i]==a[j]) {j=i;} else {j=j;} } if(j==i) printf(",%d",a[i]); } return ...`

## Low lexical similarity but same semantics

- `problem_100` token Jaccard `0.1518`: `test_9553` / `test_9811`. `void main() { char str[300]; int i,n,flag=0; int a1=0,b1=0,c1=0,d1=0,e1=0,f1=0,g1=0,h1=0,i1=0,j1=0,k1=0,l1=0,m1=0,n1=0,o1=0,p1=0,q1=0,r1=0,s1=0,t1=0,u1=0,v1=0,w1=0,x1=0,y1=0,z1=0; int a2=0,b2=0,c2=0,d2=0,e2=0,f2=0,g2=0,h...` / `void main() { char s[300],a[300]; gets(s); int i,count=0; for(i=0;s[i]!='\0';i++)a[i]=0; for(i=0;s[i]!='\0';i++) { if(s[i]==97){a[1]++;count++;} if(s[i]==98){a[2]++;count++;} if(s[i]==99){a[3]++;count++;} if(s[i]==100){a...`
- `problem_101` token Jaccard `0.1552`: `test_10203` / `test_10491`. `main() { char a,b,c,A,B,C; a=C; b=A; c=B; printf("BCA"); getchar(); }` / `/** * @file 3.cpp * @author ??? * @date 2011-10-15 * @description * ??????: ??? */ int main() { int eA, eB, eC, sA, sB, sC; // ????????????eA eB eC = 1, 2, 3????? for(eA = 1; eA <= 3; eA++) for(eB = 1; eB <= 3; eB++) for...`
- `problem_97` token Jaccard `0.1579`: `test_8060` / `test_8482`. `// // main.cpp // hw // // Created by ? ? on 12-12-16. // Copyright (c) 2012? ? ?. All rights reserved. // int main(int argc, const char * argv[]) { int num; cin>>num; int money[6]={100,50,20,10,5,1}; for (int i=0; i<6; ...` / `int main() { int n; scanf("%d",&n); if(n==42) {printf("0\n0\n2\n0\n0\n2");} else if(n==468){printf("4\n1\n0\n1\n1\n3");} else if(n==335){printf("3\n0\n1\n1\n1\n0");} else if(n==501){printf("5\n0\n0\n0\n0\n1");} else if(n...`

## Long code snippets truncated by max length

- `test_9897` `problem_100` has `2838` tokens vs max_length `512`: `void main() { char a[300]; char b[52]; int i=0,j=0; int p,q,t,s; scanf("%s",a); while(j<52) { b[j]=0; j++; } while(a[i]!='\0') { if (a[i]=='a') b[0]=b[0]+1; else if (a[i]=='b') b[1]=b[1]+1; else if (a[i]=='c') b[2]=b[2]+...`
- `test_9948` `problem_100` has `2465` tokens vs max_length `512`: `int main(){ char zfc[301]; gets(zfc); int n=strlen(zfc),c=0; char dx[27]="ABCDEFGHIJKLMNOPQRSTUVWXYZ"; char xx[27]="abcdefghijklmnopqrstuvwxyz"; int dxs[26],xxs[26],i; for(i=0;i<26;i++){ dxs[i]=0; xxs[i]=0; } char*ps; fo...`
- `test_9639` `problem_100` has `2419` tokens vs max_length `512`: `void main() { int x[52]={0},i,flag,len; char str[MAX]; scanf("%s",str); len=strlen(str); for(i=0;i<len;i++) { if(str[i]=='a') x[0]+=1; else if(str[i]=='b') x[1]+=1; else if(str[i]=='c') x[2]+=1; else if(str[i]=='d') x[3]...`
- `test_9553` `problem_100` has `1935` tokens vs max_length `512`: `void main() { char str[300]; int i,n,flag=0; int a1=0,b1=0,c1=0,d1=0,e1=0,f1=0,g1=0,h1=0,i1=0,j1=0,k1=0,l1=0,m1=0,n1=0,o1=0,p1=0,q1=0,r1=0,s1=0,t1=0,u1=0,v1=0,w1=0,x1=0,y1=0,z1=0; int a2=0,b2=0,c2=0,d2=0,e2=0,f2=0,g2=0,h...`
- `test_9665` `problem_100` has `1607` tokens vs max_length `512`: `int main() { char str[302],*ps; int i,count[26]; for(i=0;i<26;i++){ count[i]=0; } scanf("%s",str); for(ps=str;*ps!='\0';ps++){ if(*ps=='a'){ count[0]++; } } if(count[0]!=0){ printf("a=%d\n",count[0]); } for(ps=str;*ps!='...`

## Different algorithmic strategies for the same problem

- `problem_100` low-overlap same-label pair `test_9553` / `test_9811` has Jaccard `0.1518`, suggesting different implementation strategy.
- `problem_101` low-overlap same-label pair `test_10203` / `test_10491` has Jaccard `0.1552`, suggesting different implementation strategy.
- `problem_97` low-overlap same-label pair `test_8060` / `test_8482` has Jaccard `0.1579`, suggesting different implementation strategy.
- `problem_101` low-overlap same-label pair `test_10203` / `test_10234` has Jaccard `0.1579`, suggesting different implementation strategy.
- `problem_97` low-overlap same-label pair `test_8159` / `test_8482` has Jaccard `0.1639`, suggesting different implementation strategy.

## Cases where TF-IDF succeeds but neural models fail

- Pair `pair_00002198` `problem_86`: GraphCodeBERT false negative score `0.1865` despite token Jaccard `0.7609`, a case lexical matching would likely keep close.
- Pair `pair_00000806` `problem_83`: GraphCodeBERT false negative score `0.1475` despite token Jaccard `0.7333`, a case lexical matching would likely keep close.
- Pair `pair_00000542` `problem_82`: GraphCodeBERT false negative score `0.4873` despite token Jaccard `0.7273`, a case lexical matching would likely keep close.
- Pair `pair_00009518` `problem_104`: GraphCodeBERT false negative score `0.0944` despite token Jaccard `0.7255`, a case lexical matching would likely keep close.
- Pair `pair_00001336` `problem_84`: GraphCodeBERT false negative score `0.2446` despite token Jaccard `0.7179`, a case lexical matching would likely keep close.

## Cases where neural models succeed but TF-IDF fails

- Query `test_1` `problem_81`: hybrid top1 `test_1` correct score `0.9992`, TF-IDF top1 `test_1825` `problem_84`.
- Query `test_22` `problem_81`: hybrid top1 `test_22` correct score `0.9991`, TF-IDF top1 `test_775` `problem_82`.
- Query `test_35` `problem_81`: hybrid top1 `test_35` correct score `0.9989`, TF-IDF top1 `test_929` `problem_82`.
- Query `test_48` `problem_81`: hybrid top1 `test_48` correct score `0.9991`, TF-IDF top1 `test_2472` `problem_85`.
- Query `test_64` `problem_81`: hybrid top1 `test_64` correct score `0.9991`, TF-IDF top1 `test_1748` `problem_84`.
