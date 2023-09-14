import os
import torch
import torch.nn as nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel, DecisionTransformerGPT2Model
import torch
from typing import Tuple, Union
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput
import math
from pathlib import Path


class WorldModelDecisionTransformerModel(DecisionTransformerModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(config.hidden_size, 1)

        self.post_init()

    def forward(
        self,
        states=None,
        actions=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ) -> Union[Tuple, DecisionTransformerOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        stacked_inputs = (
            torch.stack((state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )
        device = stacked_inputs.device
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = self.predict_state(x[:, 1])  # predict next state given state and action
        if self.config.data_type == "delta":
            state_preds = state_preds + states

        action_preds = None
        return_preds = None

        if not return_dict:
            return (state_preds, action_preds, return_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

state_size = 6
action_time_dim = 2
action_size = 3
pred_hidden_size = 512
n_head = 1
data_type = 'delta'
model_config = DecisionTransformerConfig(state_dim=state_size+action_time_dim,
                                         act_dim=action_size,
                                         max_ep_len=2**16,
                                         hidden_size=pred_hidden_size,
                                         n_head=n_head,
                                         data_type=data_type)
model = WorldModelDecisionTransformerModel(model_config)

saved_pth = torch.load(os.path.join(Path(__file__).parent.resolve(), 'world_model_.pth'))
model.load_state_dict(saved_pth['model_state'])
pos_min_max_dic = {'boom': [-math.pi, math.pi], 'arm': [-math.pi, math.pi],
                        'bucket': [-math.pi, math.pi], 'swing': [-math.pi, math.pi]}
vel_min_max_dic = {'boom': [-0.4, 0.5], 'arm': [-0.55, 0.65],
                        'bucket': [-1.0, 1.0], 'swing': [-0.82, 0.82]}
pwm_min_max = {'boom': {"negative": [-800, -250], "positive": [250, 800]},
                    'arm': {"negative": [-800, -250], "positive": [250, 800]},
                    'bucket': {"negative": [-600, -250], "positive": [250, 600]},
                    'swing': {"negative": [-450, -180], "positive": [180, 450]}}

# 函数格式
def state_predict(states:torch.Tensor, actions:torch.Tensor, timesteps:torch.Tensor, masks:torch.Tensor, device, state_columns, action_columns) -> torch.Tensor:
    
    """
    
    输入状态时间序列(序列长度为20), 预测下一个时刻(t=21)的state;
    注意:输入输出均为原始状态

    Args:
        states (torch.tensor): 状态
        actions (torch.tensor): 动作
        timesteps (torch.tensor): 时间步
        masks (torch.tensor): 掩码
        state_columns: state中每一个维度的物理含义
        action_columns: action中每一个维度的物理含义
        device
    return:
        next_states(torch.tensor), 请使用float32类型的tensor
        
    example usage:
    
    states = torch.ones(batch_size, seq_length, state_dim)
    actions = torch.ones(batch_size, seq_length, action_dim)
    timesteps = torch.ones(batch_size, seq_length)
    masks = torch.ones(batch_size, seq_length)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_columns = state_columns = ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time']
    action_columns = ['pwm_boom', 'pwm_arm', 'pwm_swing']    
    next_state = state_predict(states, actions,timesteps, masks, device, state_columns, action_columns) -> torch.tensor(batch_size, 1, state_dim)

    """
    # note: 由于输入的是数据集中的原始数据，因此如果在训练过程中对数据做了归一化处理，则需要在该函数中对数据做归一化处理
    
    max_seq_length = 20
    
    for index, key in enumerate(state_columns):
        value = states[:,:,index]
        type, joint = key.split('_')
        if type == 'pos':
            states[:,:,index] = (value - pos_min_max_dic[joint][0]) / (pos_min_max_dic[joint][1] - pos_min_max_dic[joint][0])
        elif type == 'vel':
            states[:,:,index] = (value - vel_min_max_dic[joint][0]) / (vel_min_max_dic[joint][1] - vel_min_max_dic[joint][0])
    for index, key in enumerate(action_columns):
        value = actions[:,:,index]
        _, joint = key.split('_')
        actions[:,:,index] = ((value - pwm_min_max[joint]["positive"][0]) / (pwm_min_max[joint]["positive"][1] - pwm_min_max[joint]["positive"][0]) * (value > 0)
                            + (value - pwm_min_max[joint]["negative"][0]) / (pwm_min_max[joint]["negative"][1] - pwm_min_max[joint]["negative"][0]) * (value < 0))
        
    states = states[:, -max_seq_length:,:].to(device)
    actions = actions[:, -max_seq_length:,:].to(device)
    timesteps = timesteps[:, -max_seq_length:].to(device)
    masks = masks[:, -max_seq_length:].to(device)
    model.to(device)
    model_output = model(states, actions, timesteps, masks)
    next_state = model_output['state_preds'][:,-1,:].unsqueeze(1)
    
    # # 如果acion都为0， 上一秒的pos_state和下一秒的pos_state不变
    # bool_zero_action_filter = (actions==0)
    # inv_bool_zero_action_filter = ~bool_zero_action_filter
    # filtered_next_states_pos = bool_zero_action_filter.long()*states[:,:,:3] + inv_bool_zero_action_filter.long()*next_states[:,:,:3]
    # next_states[:,:,:3] = filtered_next_states_pos

    
    # 如果输出的是归一化后的数据，则需要做逆归一化的操作
    for index, key in enumerate(state_columns):
        value = next_state[:,:,index]
        type, joint = key.split('_')
        if type == 'pos':
            next_state[:,:,index] = value * (pos_min_max_dic[joint][1] - pos_min_max_dic[joint][0]) + pos_min_max_dic[joint][0]
        elif type == 'vel':
            next_state[:,:,index] = value * (vel_min_max_dic[joint][1] - vel_min_max_dic[joint][0]) + vel_min_max_dic[joint][0]
            
    return next_state

