# ada-job-shop

## TBD

* <img src="https://render.githubusercontent.com/render/math?math=T"> 的範圍縮小 (先用 greedy 拿 local minimum 之類的)

## Sets

<img src="https://render.githubusercontent.com/render/math?math=L">, 可用資源集合

<img src="https://render.githubusercontent.com/render/math?math=N">, 排程數 (jobs) 集合

<img src="https://render.githubusercontent.com/render/math?math=M_n,n\in N">, 在排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 時的操作數 (operations)

## Parameters

<img src="https://render.githubusercontent.com/render/math?math=W_n,n\in N">, 排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 的權重

<img src="https://render.githubusercontent.com/render/math?math=S_{nm},n\in N,m\in M_n">, 排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 操作 <img src="https://render.githubusercontent.com/render/math?math=m"> 所需資源數

<img src="https://render.githubusercontent.com/render/math?math=D_{nm},n\in N,m\in M_n">, 排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 操作 <img src="https://render.githubusercontent.com/render/math?math=m"> 操作時長

<img src="https://render.githubusercontent.com/render/math?math=P_{nm},n\in N,m\in M_n">, 排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 操作 <img src="https://render.githubusercontent.com/render/math?math=m"> 的前置條件數

<img src="https://render.githubusercontent.com/render/math?math=A_{nmp},n\in N,m\in M_n,p\in P_{nm}">, 排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 操作 <img src="https://render.githubusercontent.com/render/math?math=m"> 條件 <img src="https://render.githubusercontent.com/render/math?math=p"> 的 index

<img src="https://render.githubusercontent.com/render/math?math=T=\sum D_{nm}">, 最大時程上限 (**待優化**)

## Decision Variables

<img src="https://render.githubusercontent.com/render/math?math=y_{nmt}">, 排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 操作 <img src="https://render.githubusercontent.com/render/math?math=m"> 是否在時間 <img src="https://render.githubusercontent.com/render/math?math=t"> 使用資源, 是則為 <img src="https://render.githubusercontent.com/render/math?math=1">，否則為 <img src="https://render.githubusercontent.com/render/math?math=0">

#### 衍伸

<img src="https://render.githubusercontent.com/render/math?math=C_n=\underset{m\in M_n}{\max}\{\text{last y is true}%2BD_{nm}\},n\in N">, 排程 <img src="https://render.githubusercontent.com/render/math?math=n"> 的結束時間

<img src="https://render.githubusercontent.com/render/math?math=C_{max}=\underset{n}{\max}\{C_n\}">

## Objective

<img src="https://render.githubusercontent.com/render/math?math=\min\{C_{max}%2B\underset{i}{\sum}w_iC_i\}">

## Constraints

### Makespan 的計算

<img src="https://render.githubusercontent.com/render/math?math=C_{max}\geq C_n,\forall n">

<img src="https://render.githubusercontent.com/render/math?math=C_n\geq y_{nmt}\times t,\forall n,\forall m\in M_n,\forall t">


#### 同時間資源使用不能超過總資源數

<img src="https://render.githubusercontent.com/render/math?math=\underset{n\in N,m\in M_n}{\sum} S_{nm}\times y_{nmt}\leq |L|,\forall t\in T">

#### 操作必須連續執行(*)

<img src="https://render.githubusercontent.com/render/math?math=\sum_{i=0}^{T-1}y_{nmi}y_{nm(i%2B1)}=D_{nm}-1,\forall n,\forall m\in M_n">

#### 操作時長等於所需時長

<img src="https://render.githubusercontent.com/render/math?math=\underset{t\in T}{\sum}y_{nmt}=D_{nm},\forall n,\forall m\in M_n">

#### Dependency 限制

<img src="https://render.githubusercontent.com/render/math?math=P(\underset{t\in T}{\sum}y_{nmt}\times t, D_{nm}) - 1\geq Q(\underset{t\in T}{\sum}y_{nm't}\times t, D_{nm}),\forall m'\in A_{nmp},\forall n,\forall m\in M_n">

```python
def P(area, height):
    '''梯形上底'''
    return (area * 2 / height + height - 1) / 2
def Q(area, height):
    '''梯形下底'''
    return (area * 2 / height + 1 - height) / 2
```


#### 





p.s. 權重是浮點數，其他都是整數
