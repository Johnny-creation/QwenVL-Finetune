> 本模块负责多图多轮语境下Qwen-VL系列（不包括最新开源的Qwen3-VL-235B）微调前后模型的评估，主要代码包括回答模型生成、打分模型评估两部分。

## 评估模块结构

```bash
tree /F

│  README.md
│
├─model_generation
│      QLoRA-Qwen-2.5-VL-3B_gen_ans.py
│
└─scores
        qwen2.5_vl_3b_prompt.py
        README.md
        statistic.py
```

## 环境配置

```bash
# 创建虚拟环境并激活
conda create -n myenv python=3.10 -y
conda activate myenv

# 永久配置清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装N卡CUDA版本的PyTorch库,128适配CUDA12.8，可结合实际情况调整
pip install torch  --index-url https://download.pytorch.org/whl/cu128

pip install transformers qwen_vl_utils
pip install peft
pip install huggingface-hub modelscope
pip install tqdm matplotlib
```



## 微调&评分模型部署

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-INstruct \
--local-dir=Qwen2.5-VL-Base-Answer \
--local-dir-use-symlinks False \
--resume-download
```

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-INstruct \
--local-dir=Qwen2.5-VL-Judge \
--local-dir-use-symlinks False \
--resume-download
```

*QLoRA微调权重在`QLoRA-Qwen-2.5-VL-3B_gen_ans.py`代码中自动从ModelScope网站拉取，并将Adapter参数存储到`Qwen2.5-VL-Answer`目录下*



## 输出目录说明

![image-20251027132807520](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027132807520.png)

如图所示，3b-gene对应原始模型的对话输出，3b-peft-gene对应在MMDU-45k上QLoRA微调之后的对话输出，mark目录存储评分文件。



## 评测数据集准备

1. huggingface MMDU组合（微调+评测）数据集链接：

   ### 评测数据集特性

   - 110轮对话

   - 1600个问答对

   - 422张图片

   - 平均每个对话场景15个问题，3.8张图片，6400个词汇的Ground Truth参考答案

   - 覆盖地理、艺术、电影、交通、医药、动物、社会、建筑、城市、化学等多个方面

     ![image-20251027123007932](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027123007932.png)

   >[laolao77/MMDU at main](https://huggingface.co/datasets/laolao77/MMDU/tree/main)

2. 使用命令行拉取评测数据集文件

   ```bash
   huggingface-cli download laolao77/MMDU benchmark.json mmdu_pics.zip \
   --local-dir . \
   --local-dir-use-symlinks False \
   --repo-type dataset\
   --resume-download
   ```

   

3. 本地解压缩mmdu_pics.zip之后，在model_generation目录和scores目录分别粘贴一份mmdu_pics，根目录下可删除; benchmark.json文件保持在根目录位置。

   

## 回答生成💡

调用model_generation目录下的`QLoRA-Qwen-2.5-VL-3B_gen_ans.py`文件，生成的每个json文件对应Benchmark的每个对话场景的问题和模型回复，图片以本地存储路径的文本方式嵌入对话。

![image-20251027125712739](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027125712739.png)

在代码执行的generate方法中根据硬件设备情况调整`max_new_tokens`参数，由于对话轮数平均15轮左右，不同显卡配置应根据情况灵活调整，例如：

> 对于12GB+16GB（共享）显存的GeForce RTX 5070建议设置为128或者更低。
>
> 对于32GB+24GB（共享）显存的GeForce RTX 5090建议设置为256或者更低。



#### 回答生成参考示例

![image-20251027125851193](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027125851193.png)

![image-20251027125902312](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027125902312.png)



## 智能体评分🎓 

本评估模块的智能体评分采用本地部署本地推理的方式，完全免费但是模型参数量、推理表现和推理速度一定程度上受限。**对于前一部分每个对话场景生成的json问答文件，结合对话中生成的Reference Answer和Benchmark的Ground Truth，在Scoring rules的原则指导下，对每轮问答进行打分。**

选用Qwen2.5-VL-3B-Instruct模型进行评分，调用scores目录下的`qwen2.5_vl_3b_prompt.py`文件运行，理论上生成的每个json文件对应每个对话场景，包括每轮问答的六维评分和综合得分，下面是具体的六维评分细则。



![image-20251027123924362](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027123924362.png)

![image-20251027123940977](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027123940977.png)

![image-20251027123930103](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027123930103.png)

*注意受到硬件因素的影响，实际评分计算方式作出了调整，通过等权重计算每轮问答-有效评分的平均分的方式，给出综合得分，如果一轮对话内评分模型给出了最终综合得分，按照6倍权重进行计算，但需要注意的是，实际测评中即使上下文长度不做约束也往往不能给出。*

#### 评分生成参考示例

![image-20251027130109477](C:\Users\wf200\AppData\Roaming\Typora\typora-user-images\image-20251027130109477.png)

#### 高级功能🎯

> `statistics.py`针对多个对话场景进行综合得分汇总的计算，可以给出更权威更可靠的评分判断。



