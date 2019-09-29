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


def postprocess_raw_net_output(flow_pred, example):
    # Undo pre-processing of flow.
    flow_pred = flow_pred.squeeze(0).cpu().numpy()  # shape (2, H, H)
    flow_pred += 0.5
    flow_pred = flow_pred.transpose([1, 2, 0])

    return flow_pred


def test_flownet(net, example):
    """
    Args:
        net (FlowNet): Instance of networks.flownet.FlowNet model.
        example (dict): Un-processed example.
    """
    flow_pred = raw_net_output(net, example)
    flow_pred = postprocess_raw_net_output(flow_pred, example)

    return flow_pred
