<br>
<p align="center">
<h1 align="center"><strong>From Noise to Intent: Anchoring Generative VLA Policies with Residual Bridges</strong></h1>
  <p align="center">
      <strong><span style="color: red;"> ICML 2026</span></strong>
    <br>
   <a href="https://github.com/ymzhong66" target="_blank">Yiming Zhong*</a>&emsp;
   <a href="https://eniverz.github.io/profile" target="_blank">Yaoyu He*</a>&emsp;
   <a href="https://yizhifengyeyzm.github.io/" target="_blank">Zemin Yang*</a>&emsp;
   <a href="https://huubgit.github.io/" target="_blank">Pengfei Tian</a>&emsp;
   <a href="https://hy-van.github.io/" target="_blank">Yifan Huang</a>&emsp;
   <a href="https://openreview.net/profile?id=~Qingqiu_Huang2" target="_blank">Qingqiu Huang</a>&emsp;
   <a href="https://xingezhu.me/aboutme.html" target="_blank">Xinge Zhu</a>&emsp;
   <a href="https://yuexinma.me" target="_blank">Yuexin Ma</a>&emsp;
    <br>
    ShanghaiTech University, Morphic Robotics, The Chinese University of Hong Kong
    <br>
    *Indicates Equal Contribution
    <br>
  </p>
</p>

<p align="center">
  <a href="https://res-vla.github.io/ResVLA/"><b>📖 Project Page</b></a>
</p>
<p align="center">
  <strong>Code and Checkpoints:</strong>
  <a href="https://huggingface.co/GaussionZhong/resvla_libero_all_2B">LIBERO checkpoint</a>
  |
  <a href="https://huggingface.co/GaussionZhong/resvla_simpler_env_2B">SimplerEnv checkpoint</a>
</p>
</div>

# Environment Setup

```
git clone https://github.com/4DVLab/ResVLA.git
cd ResVLA

conda create -n resvla python=3.10 -y
conda activate resvla

pip install -r requirements.txt
pip install flash-attn --no-build-isolation

pip install -e .
```

# Train Recipes
