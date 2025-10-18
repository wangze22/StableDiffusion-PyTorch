import onnx
import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
import numpy as np
from onnx import numpy_helper


def get_torch_layer_info(torch_layer):
    """
    Get the operator type, parameter information, and weight values for a given Torch layer.
    """
    layer_type = type(torch_layer).__name__
    params = {}
    weight = None

    if hasattr(torch_layer, 'weight') and torch_layer.weight is not None:
        weight = torch_layer.weight.detach().cpu().numpy()
        params['weight_shape'] = torch_layer.weight.shape

    if hasattr(torch_layer, 'bias') and torch_layer.bias is not None:
        params['bias_shape'] = torch_layer.bias.shape

    if hasattr(torch_layer, 'stride'):
        params['stride'] = torch_layer.stride

    if hasattr(torch_layer, 'padding'):
        params['padding'] = torch_layer.padding

    if hasattr(torch_layer, 'dilation'):
        params['dilation'] = torch_layer.dilation

    if hasattr(torch_layer, 'groups'):
        params['groups'] = torch_layer.groups

    return layer_type, params, weight



def compute_node_output(node, model):
    """
    Compute the output for a given node based on its type and inputs.
    """
    input_tensors = [find_weight(inp, model) for inp in node.input]

    # Handle addition operation
    if node.op_type == 'Add':
        return np.add(*input_tensors)

    # Handle multiplication operation
    elif node.op_type == 'Mul':
        return np.multiply(*input_tensors)

    # Handle matrix multiplication operation
    elif node.op_type == 'Gemm':
        # Assuming inputs are A, B, and C where C is optional
        result = np.dot(input_tensors[0], input_tensors[1])
        if len(input_tensors) > 2:
            result += input_tensors[2]  # Adding the bias term if it exists
        return result

    # Handle transpose operation
    elif node.op_type == 'Transpose':
        # You might need to retrieve 'perm' attribute from the node to get the correct order
        perm = [i for i in node.attribute if i.name == 'perm'][0].ints if any(i.name == 'perm' for i in node.attribute) else None
        if perm is not None:
            return np.transpose(input_tensors[0], axes=tuple(perm))
        return np.transpose(input_tensors[0])

    # Handle reshape operation
    elif node.op_type == 'Reshape':
        # Shape might be given as a second tensor or as an attribute
        if len(input_tensors) > 1:
            return np.reshape(input_tensors[0], newshape=input_tensors[1])
        else:
            shape = [i for i in node.attribute if i.name == 'shape'][0].ints
            return np.reshape(input_tensors[0], newshape=tuple(shape))

    return None  # If operation type not recognized or handled


def find_weight(node_name, model):
    """
    Recursively find and compute the tensor corresponding to a given node name.
    """
    # Check if it's directly an initializer
    for initializer in model.graph.initializer:
        if initializer.name == node_name:
            return numpy_helper.to_array(initializer)

    # If not found in initializers, find the node that produces this tensor
    for node in model.graph.node:
        if node_name in node.output:
            # Compute the output based on this node's operation
            weight = compute_node_output(node, model)
            return weight

    return None  # If no computation or data found


def get_onnx_layer_info(onnx_node, onnx_model):
    """
    Get the operator type, parameter information, and weight values for a given ONNX node.
    """
    layer_type = onnx_node.op_type
    params = {}
    for attr in onnx_node.attribute:
        if attr.name == 'strides':
            params['stride'] = tuple(attr.ints)
        elif attr.name == 'pads':
            params['padding'] = tuple(attr.ints[:2])  # Assuming 2D padding
        elif attr.name == 'dilations':
            params['dilation'] = tuple(attr.ints)
        elif attr.name == 'group':
            params['groups'] = attr.i

    weight = find_weight(onnx_node.input[1], onnx_model)  # Search for the weight

    if weight is not None:
        if layer_type == 'MatMul':
            weight = weight.transpose(1, 0)
        params['weight_shape'] = weight.shape

    return layer_type, params, weight

def compare_layers(torch_layer_info, onnx_layer_info):
    """
    Compare the Torch layer information with an ONNX node.
    """
    torch_type, torch_params, torch_weight = torch_layer_info
    onnx_type, onnx_params, onnx_weight = onnx_layer_info

    type_mapping = {
        'Conv2d': 'Conv',
        'Conv2d_cim': 'Conv',
        'Linear': 'Gemm',
        'Linear_cim': 'Gemm',
        'MatMul': 'Gemm',
        'Conv2d_quant_noise': 'Conv',
        'Linear_quant_noise': 'Gemm',
        'Conv2d_qn_lsq': 'Conv',
        'Linear_qn_lsq': 'Gemm',
        'Conv2d_ADDA_aware': 'Conv',
        'Linear_ADDA_aware': 'Gemm',
        # Add more mappings if needed
    }

    if type_mapping.get(torch_type, torch_type) != type_mapping.get(onnx_type, onnx_type):
        return False

    if onnx_params.get('weight_shape') != torch_params.get('weight_shape'):
        return False

    if not np.allclose(onnx_weight, torch_weight, rtol = 1e-04, atol = 1e-07):
        return False

    for param in ['stride', 'padding', 'dilation', 'groups']:
        if torch_params.get(param) != onnx_params.get(param):
            return False


    return True


def get_onnx_layer_name(torch_layer, onnx_model):
    """
    Get the corresponding ONNX layer name for a given Torch layer.
    """
    torch_layer_info = get_torch_layer_info(torch_layer)

    for node in onnx_model.graph.node:
        if node.op_type in ['Conv', 'MatMul', 'Gemm']:
            onnx_layer_info = get_onnx_layer_info(node, onnx_model)
            if compare_layers(torch_layer_info, onnx_layer_info):
                return node.name
    return None
