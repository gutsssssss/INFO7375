import math_functions as mf

# define the size, classes number and data address
img_height = 20
img_width = 20
size_input = img_width * img_height
num_classes = 10
batch_size = 10
address1 = 'generated_images/*'
address2 = 'generated_images2/*'
# 初始学习率
initial_lr = 0.00001
# 衰减因子
decay_rate = 0.001
# epoch数
epochs = 10
# 正则化参数
lamda = 0.5

normalization = 'L2'
