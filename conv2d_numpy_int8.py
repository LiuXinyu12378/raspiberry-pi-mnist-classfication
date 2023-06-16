import numpy as np
import os
from PIL import Image


def conv2d(images,weight,bias,stride=1,padding=1):
    # 卷积操作
    N, C, H, W = images.shape
    F, _, HH, WW = weight.shape
    # 计算卷积后的输出尺寸
    H_out = (H - HH + 2 * padding) // stride + 1
    W_out = (W - WW + 2 * padding) // stride + 1
    # 初始化卷积层输出
    out = np.zeros((N, F, H_out, W_out))
    # 执行卷积运算
    for i in range(H_out):
        for j in range(W_out):
            # 提取当前卷积窗口
            window = images[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            # 执行卷积运算
            out[:, :, i, j] = np.sum(window * weight, axis=(1, 2, 3)) + bias
    # 输出结果
    # print("卷积层输出尺寸:", out.shape)
    return out.astype(np.int16)


def max_pool2d(input, kernel_size, stride=None, padding=0):
    # 输入尺寸
    batch_size,num_channels, input_height, input_width = input.shape

    # 默认stride等于kernel_size
    if stride is None:
        stride = kernel_size

    # 输出尺寸
    output_height = (input_height - kernel_size + 2 * padding) // stride + 1
    output_width = (input_width - kernel_size + 2 * padding) // stride + 1

    # 添加padding
    if padding > 0:
        padded_input = np.pad(input, [(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='constant')
    else:
        padded_input = input

    # 用于存储输出特征图
    output = np.zeros((batch_size,num_channels, output_height, output_width))

    # 执行最大池化操作
    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size
            sub_future = padded_input[:,:, start_i:end_i, start_j:end_j]
            output[:, :,i, j] = np.max(sub_future, axis=(2,3))

    return output


def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    '''
    把输入数据映射到量化比例上
    '''
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale

    q_x = np.clip(q_x, qmin, qmax)
    q_x = np.round(q_x)   #clamp 超过最大值最小值的部分使用qmin和qmax
    
    return q_x.astype(np.int8)

# 加载保存的模型数据
model_data = np.load('conv2d_quantization_model.npz')
# 提取模型参数
conv_weight = model_data['conv1.weight'].astype(np.int8)
conv_bias = model_data['conv1.bias'].astype(np.int8)
fc_weight = model_data['fc.weight'].astype(np.int8)
fc_bias = model_data['fc.bias'].astype(np.int8)

image_score = model_data['qconv1.qi.scale']
image_zero_point = model_data['qconv1.qi.zero_point'].astype(np.int8)


for i in model_data:
    print(i)
    

# 进行推理
def inference(images):
    images = quantize_tensor(images,image_score,image_zero_point)
    images = images - image_zero_point
    # 执行卷积操作
    conv_output = conv2d(images, conv_weight, conv_bias, stride=1, padding=0)

    conv_output = model_data["qconv1.M"] * conv_output
    conv_output = np.round(conv_output) 
    conv_output = conv_output+model_data["qconv1.qo.zero_point"].astype(np.int8)
    conv_output = np.clip(conv_output, 0, 2.**8-1.)
    conv_output = np.round(conv_output) 
    conv_output = conv_output.astype(np.int8)
    
    conv_output = np.maximum(conv_output, 0)  # ReLU激活函数
    #maxpool2d
    pool = max_pool2d(conv_output,2)
    # 执行全连接操作
    flattened = pool.reshape(pool.shape[0], -1)
    flattened = flattened - model_data["qfc.qi.zero_point"].astype(np.int8)
    flattened = flattened.astype(np.int8)
    fc_output = np.dot(flattened, fc_weight.T) + fc_bias
 
    fc_output = fc_output * model_data["qfc.M"]
    fc_output = np.round(fc_output)
    fc_output = fc_output + model_data["qfc.qo.zero_point"].astype(np.int8)
    fc_output = np.clip(fc_output, 0, 2.**8-1.)
    fc_output = np.round(fc_output) 
    
    # 获取预测结果
    predictions = np.argmax(fc_output, axis=1)

    return predictions



def infer_images_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(file_path)
            label = file_name.split(".")[0].split("_")[1]
            image = np.array(image)/255.0
            image = np.expand_dims(image,axis=0)
            image = np.expand_dims(image,axis=0)
            
            predicted_class = inference(image)
            print("file_path:",file_path,"img size:",image.shape,"label:",label,'Predicted class:', predicted_class)




if __name__ == "__main__":

    # conv2d(X,weight,b)
    folder_path = './mnist_pi'  # 替换为图片所在的文件夹路径
    infer_images_in_folder(folder_path)