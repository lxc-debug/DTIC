from model.attention_model import MainSparseModel
import torch
import random
import numpy as np
from onnx2torch import convert
import dgl
import onnx
from tqdm import tqdm
import copy

device=torch.device('cuda:%d'%0)
num_nodes = 650
bin_num = 14
op_type_size = 215
hidden_dim = 32
row_size = 4096
with open('op_list.txt', mode='r', encoding='utf16') as fp:
    content = fp.read()
    op_li = content.split(',')



def pos_emb(
    pos_array: np.ndarray, pos_emb_dim: int,
):

    freq_band = [1 / (10000 ** (idx / pos_emb_dim))
                    for idx in range(pos_emb_dim)]
    fn_list = []
    op_fn = [np.sin, np.cos]
    for freq in freq_band:
        for op in op_fn:
            fn_list.append(lambda x, freq=freq, op=op: op(x * freq))

    return np.concatenate(
        [np.expand_dims(fn(pos_array), -1) for fn in fn_list], axis=-1
    )

def onnx2dgl(
    graph,
    parameter_dict,
) -> tuple:
    # par_li = list()
    graph_par_li = list()
    row_size = list()
    node_index = dict()
    edge_src = list()
    edge_dst = list()
    node_size = len(list(graph.node))
    pos_arr = torch.zeros((node_size,),dtype=torch.float32)
    parameter_li = list()    
    
    if num_nodes < node_size:
        raise ValueError(f'node size seeting small, needing {node_size}')

    for index, node in enumerate(graph.node):
        # 首先是对参数的操作，然后是对结构的操作
        pos_arr[index] = index + 1
        arr = None
        for in_put in node.input:
            # todo 对于没有参数的层，我只给了row_size为1，主要是为了给到两维，其余在coll_fn在后面padding
            # 想了一下，这一版还是主要考虑w这个参数
            if in_put in parameter_dict.keys() and (in_put.split('.')[-1] == 'weight' and parameter_dict[in_put].ndim > 1) or (in_put.startswith('onnx::Conv') and parameter_dict[in_put].ndim == 4):
                arr = parameter_dict[in_put]
                parameter_li.append([arr, in_put])  # 后序使用的就是这个列表
                            

# todo 这个改掉一个bug，因为如果没有进入到内部循环的话，arr拿不到值，同样rowarr也没值，这样就不会加入到par_arr之中，所以就会导致拿到的值全部都是带参数的
        if arr is not None:
            row_size.append(arr.shape[0])
        else:
            row_size.append(0)
            parameter_li.append([torch.zeros((1, bin_num-1), dtype=torch.float32), None])

        # 下面是处理图的数据的,这里的+1都是为了给空节点的0留位置
        op_type = np.zeros((op_type_size,), dtype=np.float32)
        op_index = op_li.index(node.op_type)
        op_type[op_index] = 1.

        pos = pos_emb(np.array([index+1], dtype=np.float32),
                      hidden_dim//2).reshape((-1,))

        # 现在还缺的就是一个pos_emb，然后把这些东西展开拼起来，最后还有就是mask的大小还没有确定。
        struct_arr = np.concatenate((op_type,  pos),axis=-1)
        struct_tensor = torch.FloatTensor(struct_arr)
        graph_par_li.append(struct_tensor)

        # 下面是处理结构的连边关系的
        for out in node.output:  # 得到节点输出用来图的连接
            node_index[out] = index

        for in_put in node.input:
            if in_put in node_index.keys():
                edge_src.append(node_index[in_put])
                edge_dst.append(index)

    # todo 到这里了       

    dgl_graph = dgl.to_bidirected(
        dgl.graph((edge_src, edge_dst), num_nodes=node_size)) # 这里直接给定nodesize就行了

    graph_par_tensor = torch.stack(graph_par_li,dim=0)
    # par_tensor = torch.FloatTensor(par_arr)

    dgl_graph.ndata['struct_feature'] = graph_par_tensor

    return (dgl_graph, row_size, max(row_size), node_size), parameter_li

def convert_onnx_model(onnx_path:str):
    onnx_model_path = onnx_path  #'poisoned_resnet34_0_id-00000016.onnx'
    # input_shape = (1, 3, 224, 224)
    model = convert(onnx_model_path)
    model = model.to(device)
    return model

# def freeze_parameter(model:torch.nn.Module): # eval模式就是冻结参数的
#     for _, parameter in model.named_parameters():
#         parameter.requires_grad = False


def get_onnx_model(onnx_path):
    # onnx_path = 'poisoned_data/poisoned_resnet50_4_id-00000189.onnx'
    onnx_model = onnx.load(onnx_path)
    
    graph = onnx_model.graph

    parameter_dict = dict()  # 网络参数

    for par in graph.initializer:  # 转换所有的网络参数
        
        par_np = np.frombuffer(
            par.raw_data, dtype=np.float32
        ).reshape(par.dims)
        par_tensor = torch.tensor(par_np)
        par_tensor.requires_grad = False

        parameter_dict[par.name] = par_tensor

    return graph, parameter_dict, onnx_model

def get_statistic(parameter_li:list):
    par_arr = list()
    for arr_tuple in parameter_li:
        if arr_tuple[1] is None:
            par_arr.append(arr_tuple[0])
        else:
            arr = arr_tuple[0]
            if arr.ndim == 4:  # 这个是conv的参数
                row_shape, _, _, _ = arr.shape
                arr = arr.reshape((row_shape, -1))
                        

            if arr.shape[0] > row_size:
                raise ValueError(
                    f'row size is out of setting, now needing space is {arr.shape[0]}')
            
            row_arr = torch.zeros((arr.shape[0], bin_num - 1), dtype = torch.float32)

            num_max = torch.max(arr)
            num_min = torch.min(arr)
            # 把这个运算放到外面就只需要算一次了
            arr_norm = (arr - num_min) / (num_max - num_min)

            arr_mean = torch.mean(arr)
            arr_one_norm = torch.greater_equal(
                arr, arr_mean.reshape(-1, 1)).to(torch.float32)
            
            arr_var=arr.mean(axis = 1).var()
            arr_norm_var=arr_norm.mean(axis = 1).var()
            arr_one_norm_var=arr_one_norm.mean(axis = 1).var()

            for idx in range(arr.shape[0]):
                # origin
                arr_tmp = arr[idx, :]
                row_arr[idx][0] = arr_tmp.min()
                row_arr[idx][1] = arr_tmp.max()
                row_arr[idx][2] = arr_tmp.mean()
                row_arr[idx][3] = arr_tmp.var()
                row_arr[idx][4] = arr_var

                tmp = arr_norm[idx, :]
                # min max norm
                row_arr[idx][5] = tmp.min()
                row_arr[idx][6] = tmp.max()
                row_arr[idx][7] = tmp.mean()
                row_arr[idx][8] = tmp.var()
                row_arr[idx][9] = arr_norm_var

                tmp_one = arr_one_norm[idx, :]

                # zero one norm
                row_arr[idx][10] = tmp_one.mean()
                row_arr[idx][11] = tmp_one.var()
                row_arr[idx][12] = arr_one_norm_var
            
            par_arr.append(row_arr)

    return par_arr



#todo 1. 首先根据函数推断哪三层要加掩码，给出对应位置
#todo 2. 生成三个掩码，并根据对应位置，把全零矩阵加入到parameter_li中的对应元素中
#todo 3. 循环，直到最终的结果发生转变，将原始数据置零
#todo 4. 最后就是根据对应位置，修改原始graph的值，并导出为onnx文件

def get_index(model:torch.nn.Module, data_tuple:tuple, parameter_li:list):
    model.eval()
    with torch.no_grad():
        par_arr = get_statistic(parameter_li)
        data_batch = [(*data_tuple, par_arr)]
        graph, data, row_mask, node_mask = coll_fn(data_batch)
        res, attn = model(graph, data, row_mask, node_mask)
        attn_mean = torch.mean(attn, dim=-3)
        attn_mean = torch.squeeze(attn_mean)
        _, k_index = torch.topk(attn_mean, 10, dim=-1)
        k_index = k_index.tolist()
        print(k_index,attn_mean.shape)

        return k_index
    

def get_mask(parameter_li:list, k_index:torch.Tensor):
    mask_li = list()
    for idx in range(len(parameter_li)):
        if idx in k_index:
            mask = torch.zeros_like(parameter_li[idx][0],requires_grad=True)
            mask_li.append((mask, idx))

    return mask_li

def add_mask(parameter_li:list, mask_li:list):
    parameter_li_copy = copy.deepcopy(parameter_li)
    for mask, idx in mask_li:
        parameter_li_copy[idx][0] += mask

    return parameter_li_copy
        

def mitigate_onnx_model(model:torch.nn.Module, data_tuple:tuple, parameter_li:list,mask_li:list):
    model.eval()
    optimizer = torch.optim.Adam([item[0] for item in mask_li], lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(20)):
        parameter_li_copy = add_mask(parameter_li, mask_li)
        par_arr = get_statistic(parameter_li_copy)
        data_batch = [(*data_tuple, par_arr)]
        graph, data, row_mask, node_mask = coll_fn(data_batch)
        res, _ = model(graph, data, row_mask, node_mask)

        label = torch.tensor(0, dtype=torch.long, device=device)
        loss = loss_fn(res, label)
        for item in mask_li:
            loss += 8e-4 * torch.sum(torch.abs(item[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch:{epoch}, loss:{loss.item()}, res:{res.tolist()}')

    parameter_li_copy = add_mask(parameter_li, mask_li)
    return parameter_li_copy #todo 后续把这个函数写完

def save_new_onnx_model(onnx_model, parameter_li:list, new_onnx_model_path:str):

    graph = onnx_model.graph
    par_dict = dict()
    for item in parameter_li:
        par_dict[item[1]] = item[0]
    for par in graph.initializer:  # 转换所有的网络参数
        if par.name in par_dict.keys():
            data = par_dict[par.name]
            par.raw_data = data.detach().numpy().tobytes()

    onnx.save(onnx_model, new_onnx_model_path)


def coll_fn(batch):
    data_li = list()
    row_size_li = list()
    max_row_size_li = list()
    node_size_li = list()
    graph_li = list()

    for graph, row_size, max_row_size, node_size, data in batch:
        
        data_li.append(data)
        max_row_size_li.append(max_row_size)
        row_size_li.append(row_size)
        node_size_li.append(node_size)
        graph_li.append(graph)

    batch_row_size=max(max_row_size_li)
    batch_node_size=max(node_size_li)
    batch_size=len(batch)
    row_mask=list()
    node_mask=torch.zeros((batch_size,batch_node_size),dtype=torch.float32,device=device)

    for idx in range(batch_size):
        for jdx in range(len(row_size_li[idx])):
            row_mask_tmp=torch.zeros((batch_row_size,),dtype=torch.float32)
            row_mask_tmp[:row_size_li[idx][jdx]]=1.
            row_mask.append(row_mask_tmp)

        node_mask[idx][:node_size_li[idx]]=1.

    row_mask=torch.stack(row_mask,dim=0).to(device)
    row_mask=torch.unsqueeze(row_mask,dim=1)
    node_mask=torch.unsqueeze(node_mask,dim=1)  # 这里是扩展一维用于后续计算

    # todo 下面就是padding的过程了,这里可以顺便遍历的时候把mask也整了
    data_tmp=list()
    for graph in data_li:
        data_tmp.extend(graph)
    data_res=torch.nn.utils.rnn.pad_sequence(data_tmp,batch_first=True)

    return dgl.batch(graph_li).to(device), (data_res.to(device),node_size_li), row_mask, node_mask

def main_mitigate(onnx_path:str = 'poisoned_data/poisoned_id-00000001.onnx', new_onnx_model_path:str = 'poisoned_data/poisoned_id-00000001_mitigate.onnx', new_pt_model_path:str = 'poisoned_data/mitigate.pt'):
    model = MainSparseModel()
    model.load_state_dict(torch.load('best_parameter/leader_three/aug_rp_3_best.pt'))
    model = model.to(device)

    graph, parameter_dict, onnx_model = get_onnx_model(onnx_path)
    data_tuple, parameter_li = onnx2dgl(graph, parameter_dict)
    k_index = get_index(model, data_tuple, parameter_li)
    mask_li = get_mask(parameter_li, k_index)
    parameter_li_copy = mitigate_onnx_model(model, data_tuple, parameter_li, mask_li)
    save_new_onnx_model(onnx_model, parameter_li_copy, new_onnx_model_path)

    new_model = convert_onnx_model(new_onnx_model_path)
    torch.save(new_model, new_pt_model_path)


main_mitigate()
    


