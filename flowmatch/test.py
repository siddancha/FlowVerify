import torch


def raw_net_output(net, example):
    """
    Args:
        net (FlowNet): Instance of networks.flownet.FlowNet model.
        example (dict): Un-processed example.
    Returns:
        flow_pred (Tensor, shape=1x2xHxH, type=FloatTensor): Raw flow prediction output of net.
    """
    net.eval()

    example = net.preprocess(example)
    cs_input, tg_input = example['net_cs_im'], example['net_tg_im']
    cs_input, tg_input = cs_input.unsqueeze(0).cuda(), tg_input.unsqueeze(0).cuda()

    with torch.no_grad():
        flow_pred, stride = net(cs_input, tg_input)[0][0]  # 0 indexes final flow_pred and its stride
        flow_pred = net.upsample(flow_pred, scale_factor=stride)

    return flow_pred

def raw_net_output_batch(net, examples):
    """
    Args:
        net (FlowNet): Instance of networks.flownet.FlowNet model.
        example (dict): Un-processed example.
    Returns:
        flow_pred (Tensor, shape=1x2xHxH, type=FloatTensor): Raw flow prediction output of net.
    """
    net.eval()
    
    cs_inputs, tg_inputs = [], []
    for example in examples:
        example = net.preprocess(example)
        cs_input, tg_input = example['net_cs_im'], example['net_tg_im']
        cs_input, tg_input = cs_input.unsqueeze(0).cuda(), tg_input.unsqueeze(0).cuda()
        cs_inputs += [cs_input]
        tg_inputs += [tg_input]

    cs_inputs = torch.cat(cs_inputs, 0)
    tg_inputs = torch.cat(tg_inputs, 0)
    # print(cs_inputs.shape, tg_inputs.shape)

    with torch.no_grad():
        flow_pred, stride = net(cs_inputs, tg_inputs)[0][0] # 0 indexes final flow_pred and its stride
        flow_pred = net.upsample(flow_pred, scale_factor=stride)

    return flow_pred

def postprocess_raw_net_output(flow_pred, example):
    # Undo pre-processing of flow.
    flow_pred = flow_pred.squeeze(0).cpu().numpy()  # shape (2, H, H)
    flow_pred += 0.5
    flow_pred = flow_pred.transpose([1, 2, 0])

    return flow_pred

def postprocess_raw_net_output_batch(flow_preds):
    # Undo pre-processing of flow.
    results = []
    for flow_pred in flow_preds:
        flow_pred = flow_pred.squeeze(0).cpu().numpy()  # shape (2, H, H)
        flow_pred += 0.5
        flow_pred = flow_pred.transpose([1, 2, 0])
        results.append(flow_pred)
    return results

def test_flownet(net, example):
    """
    Args:
        net (FlowNet): Instance of networks.flownet.FlowNet model.
        example (dict): Un-processed example.
    """
    flow_pred = raw_net_output(net, example)
    flow_pred = postprocess_raw_net_output(flow_pred, example)

    return flow_pred
