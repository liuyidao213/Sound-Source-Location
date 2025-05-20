# Sound-Source-Location
graduation project

This project is my graduation project. The dataset used is the public SLoClas dataset, and the download path is: https://bidishasharma.github.io/SLoClass/. 
The experiments described in this paper were conducted on a Windows operating system. The hardware environment includes an 11th Gen Intel(R) Core(TM) i5-11400H CPU, an NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB). The software environment consists of Python 3.10.16, PyTorch 2.5.1, NumPy 1.24.0, SciPy 1.25.1, hdf5storage 0.1.19, and other relevant packages.
This paper uses the dataset after feature extraction. After the environment is configured, you need to enter the following code in the terminal to run:  
python sound_source_location.py -input all -wts0 1 -batch 32 -epoch 100

When debugging, enter the following code to test with a small dataset:  
python sound_source_location.py -input small -wts0 1 -batch 32 -epoch 20

If you want to learn more about this project, you can read this CSDN article: https://blog.csdn.net/TB___/article/details/147929356?sharetype=blogdetail&sharerId=147929356&sharerefer=PC&sharesource=TB___&spm=1011.2480.3001.8118
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
该项目为本人的毕业设计，数据集使用的的是SLoClas公开数据集，下载路径为：https://bidishasharma.github.io/SLoClass/。
本文所进行的实验在 windows 操作系统下进行，硬件环境为 CPU11th Gen Intel(R) Core(TM) i5-11400H，GPU为 NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB)，软件环境为 python3.10.16、pytorch2.5.1、numpy1.24.0、scripy1.25.1、hdf5storage0.1.19 等。
本文使用的是已经经过特征提取后的数据集，环境配置好后需要在终端中输入以下代码方可运行：​
python sound_source_location.py -input all -wts0 1 -batch 32 -epoch 100
调试的时候输入以下代码用小数据集测试：​
python sound_source_location.py -input small -wts0 1 -batch 32 -epoch 20

如果想要了解更多该项目，可以越多这篇CSDN文章：https://blog.csdn.net/TB___/article/details/147929356?sharetype=blogdetail&sharerId=147929356&sharerefer=PC&sharesource=TB___&spm=1011.2480.3001.8118
