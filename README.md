# 小学口算数据生成

## 使用方法
第一步：

```
cd datasets
unzip backgrd.zip ./
unzip compare_op.zip ./
unzip hand_written.zip ./ 

```

第二步：

```
# 生成20000条四则运算样本
python gen_common.py 20000 


# 生成20000条比大小样本
python gen_daxiao.py 20000 


# 生成20000条单位换算样本
python gen_unit.py 20000 
```

第三步：

生成其他题型可任意修改此代码